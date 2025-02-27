"""Model for DeformingPlate."""

import torch
from torch import nn as nn
# import torch.nn.functional as F
# from torch import  as F

from utilities import common
from models import normalization, encode_process_decode, gcn, regDGCNN_seg, regpointnet_seg

import torch_scatter
from torch_geometric.data import Data


device = torch.device('cuda')


class Model(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, params, core_model_name="encode_process_decode", message_passing_aggregator='sum',
                 message_passing_steps=15, attention=False):
        super(Model, self).__init__()
        self._params = params
        self._output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = normalization.Normalizer(size=4, name='node_normalizer')
        self._node_dynamic_normalizer = normalization.Normalizer(size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = normalization.Normalizer(size=8, name='mesh_edge_normalizer')
        self._world_edge_normalizer = normalization.Normalizer(size=4, name='world_edge_normalizer')
        self._model_type = params['model'].__name__
        self._displacement_base = None

        self.core_model_name = core_model_name
        if core_model_name == 'encode_process_decode':
            self.core_model = encode_process_decode
            self.message_passing_steps = message_passing_steps
            self.message_passing_aggregator = message_passing_aggregator
            self._attention = attention
            self.is_multigraph = True
            self.learned_model = self.core_model.EncodeProcessDecode(
                output_size=params['size'],
                latent_size=128,
                num_layers=2,
                message_passing_steps=self.message_passing_steps,
                message_passing_aggregator=self.message_passing_aggregator, attention=self._attention
            )

        elif core_model_name == 'gcn':
            self.core_model = gcn
            self.is_multigraph = False
            self.learned_model = gcn.GCN(
                nfeat=9,  # Assuming node feature size is 4
                nhid=128,
                output=params['size'],
                dropout=0.1,
                edge_dim=8  # Assuming edge feature size is 8
            )
        elif core_model_name == 'regDGCNN_seg':
            self.core_model = regDGCNN_seg
            self.is_multigraph = False
            self.learned_model = regDGCNN_seg.regDGCNN_seg(
                output_size=params['size'],
                input_dims=12,  
                k=20,  
                emb_dims=1024,  
                dropout=0.1
            )
        elif core_model_name == "regpointnet_seg":
            self.core_model = regpointnet_seg
            self.is_multigraph = False
            self.learned_model = regpointnet_seg.regpointnet_seg(
                output_size=params['size'],
                input_channel=12
            )
        else:
            raise ValueError(f"Unsupported core model: {self.core_model_name}")

    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape)
        if operation == 'sum':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result

    def _build_graph(self, inputs, is_training,multigraph=True):
        """Builds input graph."""
        world_pos = inputs['curr_pos'].to(device)
        node_type = inputs['node_type'].to(device)

        one_hot_node_type = nn.functional.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE).float()

        cells = inputs['cells'].to(device)
        decomposed_cells = common.triangles_to_edges(cells, deform=True)
        senders, receivers = decomposed_cells['two_way_connectivity']


        # find world edge
        radius = 25
        world_distance_matrix = torch.cdist(world_pos, world_pos, p=2)
        # print("----------------------------------")
        # print(torch.nonzero(world_distance_matrix).shape[0])
        world_connection_matrix = torch.where(world_distance_matrix < radius, True, False)
        # print(torch.nonzero(world_connection_matrix).shape[0])
        # remove self connection
        world_connection_matrix = world_connection_matrix.fill_diagonal_(False)
        # print(torch.nonzero(world_connection_matrix).shape[0])
        # remove world edge node pairs that already exist in mesh edge collection
        world_connection_matrix[senders, receivers] = torch.tensor(False, dtype=torch.bool, device=device)
        # only obstacle and handle node as sender and normal node as receiver
      
        # remove receivers whose node type is obstacle
        no_connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        no_connection_mask_t = torch.transpose(torch.stack([no_connection_mask] * world_pos.shape[0], dim=1), 0, 1)
        world_connection_matrix = torch.where(no_connection_mask_t, torch.tensor(False, dtype=torch.bool, device=device), world_connection_matrix)
        # remove senders whose node type is handle and normal
        connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        connection_mask = torch.stack([no_connection_mask] * world_pos.shape[0], dim=1)
        world_connection_matrix = torch.where(connection_mask, world_connection_matrix, torch.tensor(False, dtype=torch.bool, device=device))
      
        world_senders, world_receivers = torch.nonzero(world_connection_matrix, as_tuple=True)

        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=world_receivers) -
                              torch.index_select(input=world_pos, dim=0, index=world_senders))

        mesh_pos = inputs['mesh_pos'].to(device)
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, receivers))
        all_relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
                              torch.index_select(input=world_pos, dim=0, index=receivers))
        mesh_edge_features = torch.cat((
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True),
            all_relative_world_pos,
            torch.norm(all_relative_world_pos, dim=-1, keepdim=True)), dim=-1)
        if multigraph:
            world_edge_features = torch.cat((
                relative_world_pos,
                torch.norm(relative_world_pos, dim=-1, keepdim=True)), dim=-1)
            world_edges = self.core_model.EdgeSet(
                name='world_edges',
                features=self._world_edge_normalizer(world_edge_features, None, is_training),
                # features=world_edge_features,
                receivers=world_receivers,
                senders=world_senders)
            
            mesh_edges = self.core_model.EdgeSet(
                name='mesh_edges',
                features=self._mesh_edge_normalizer(mesh_edge_features, None, is_training),
                # features=mesh_edge_features,
                receivers=receivers,
                senders=senders)
            node_features =  one_hot_node_type

            return (self.core_model.MultiGraph(node_features=node_features,
                                              edge_sets=[mesh_edges, world_edges]))
        
        else:
            world_edge_features = torch.cat((
                relative_world_pos,
                torch.norm(relative_world_pos, dim=-1, keepdim=True),
                torch.zeros((relative_world_pos.shape[0], 4), device=device)  # Padding
            ), dim=-1)
            combined_senders = torch.cat([senders, world_senders], dim=0)
            combined_receivers = torch.cat([receivers, world_receivers], dim=0)
            edge_features = torch.cat([mesh_edge_features, world_edge_features], dim=0)

            edge_index = torch.stack([combined_senders, combined_receivers], dim=0)

            return Data(x=one_hot_node_type, edge_index=edge_index, edge_attr=edge_features)


    def forward(self, inputs, is_training):
        if self.core_model_name == "regDGCNN_seg" or self.core_model_name == "regpointnet_seg":
            if is_training:
                return self.learned_model(inputs) 
            else: 
                return self._update(inputs, self.learned_model(inputs))
        graph = self._build_graph(inputs, is_training=is_training,multigraph=self.is_multigraph)
        if is_training:
            return self.learned_model(graph, is_training=is_training)
        else:
            return self._update(inputs, self.learned_model(graph,  is_training=is_training))

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        '''output_mask = torch.eq(inputs['node_type'][:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
        output_mask = torch.stack([output_mask] * inputs['world_pos'].shape[-1], dim=1)
        velocity = self._output_normalizer.inverse(torch.where(output_mask, per_node_network_output, torch.tensor(0., device=device)))'''
        # print("per node network output shape",per_node_network_output.shape)
        velocity = self._output_normalizer.inverse(per_node_network_output[:, :3])

        node_type = inputs['node_type']
        '''scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)'''

        # integrate forward
        cur_position = inputs['curr_pos']
        position = cur_position + velocity
        # position = torch.where(scripted_node_mask, position + inputs['target|world_pos'] - inputs['world_pos'], position)
        return (position, cur_position, velocity)

    def get_output_normalizer(self):
        return (self._output_normalizer)

    def save_model(self, path):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._output_normalizer, path + "_output_normalizer.pth")
        torch.save(self._node_dynamic_normalizer, path + "_node_dynamic_normalizer.pth")
        # torch.save(self._stress_output_normalizer, path + "_stress_output_normalizer.pth")
        torch.save(self._mesh_edge_normalizer, path + "_mesh_edge_normalizer.pth")
        torch.save(self._world_edge_normalizer, path + "_world_edge_normalizer.pth")
        torch.save(self._node_normalizer, path + "_node_normalizer.pth")

    def load_model(self, path):
        self.learned_model = torch.load(path + "_learned_model.pth")
        self._output_normalizer = torch.load(path + "_output_normalizer.pth")
        self._node_dynamic_normalizer = torch.load(path + "_node_dynamic_normalizer.pth")
        self._stress_output_normalizer = torch.load(path + "_stress_output_normalizer.pth")
        self._mesh_edge_normalizer = torch.load(path + "_mesh_edge_normalizer.pth")
        self._world_edge_normalizer = torch.load(path + "_world_edge_normalizer.pth")
        self._node_normalizer = torch.load(path + "_node_normalizer.pth")

    def evaluate(self):
        self.eval()
        self.learned_model.eval()
