import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.GCN_layers import GraphConvolution
from torch_geometric.nn import GraphConv, TopKPooling,  GINEConv, avg_pool, TAGConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# from models.GCN_utils import build_graph
from utilities import common
from models import normalization

device = torch.device('cuda')

def build_graph(output_normalizer,world_edge_normalizer,trajectory, k=20):
    
    node_type = trajectory['node_type'].to(device)  # Types for nodes (if needed, can be used for labeling or grouping)
    one_hot_node_type = nn.functional.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE).float()
    world_pos = trajectory['curr_pos'].to(device)  # Positional data (used for features)
    target_world_pos = trajectory['next_pos'].to(device)  # Target world position (for loss calculation)
    mesh_pos = trajectory['mesh_pos'].to(device)
    cells = trajectory['cells'].to(device)
    #cur_position = world_pos
    target_position = target_world_pos
    target_velocity = target_position - mesh_pos
    target = output_normalizer(target_velocity)
    decomposed_cells = common.triangles_to_edges(cells, deform=True)
    senders, receivers = decomposed_cells['two_way_connectivity']
    print('senders, receivers:',senders.shape, receivers.shape)
    radius = 25

    world_distance_matrix = torch.cdist(world_pos, world_pos, p=2)
    world_connection_matrix = torch.where(world_distance_matrix < radius, True, False)
    world_connection_matrix = world_connection_matrix.fill_diagonal_(False)
    world_connection_matrix[senders, receivers] = torch.tensor(False, dtype=torch.bool, device=device)

    no_connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
    no_connection_mask_t = torch.transpose(torch.stack([no_connection_mask] * world_pos.shape[0], dim=1), 0, 1)
    world_connection_matrix = torch.where(no_connection_mask_t, torch.tensor(False, dtype=torch.bool, device=device), world_connection_matrix)
    # remove senders whose node type is handle and normal
    connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
    connection_mask = torch.stack([no_connection_mask] * world_pos.shape[0], dim=1)
    world_connection_matrix = torch.where(connection_mask, world_connection_matrix, torch.tensor(False, dtype=torch.bool, device=device))

    world_senders, world_receivers = torch.nonzero(world_connection_matrix, as_tuple=True)
    print('Shape of senders and receivers resp:',world_senders.shape, world_receivers.shape)
    # Find world edge indices (i.e., edges that connect senders and receivers)
    edge_index = torch.stack((world_senders, world_receivers), dim=0).to(device)
    # Calculate relative positions for edge features
    relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=world_receivers) -
                          torch.index_select(input=world_pos, dim=0, index=world_senders))

    # Compute edge features (e.g., relative position, distance)
    world_edge_features = torch.cat((
        relative_world_pos,
        torch.norm(relative_world_pos, dim=-1, keepdim=True)), dim=-1)
    
    world_edge_features = world_edge_normalizer(world_edge_features)
    node_features = one_hot_node_type

    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    world_edge_features = world_edge_features.to(device)

    target = torch.FloatTensor(target.cpu().numpy()).to(device)
    #adj = torch.FloatTensor(np.array(adj.todense()))  # Ensure adjacency matrix is a tensor
    return node_features, edge_index, world_edge_features, target


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, output, dropout, edge_dim):
        super(GCN, self).__init__()
        '''
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.out = nn.Linear(nhid, output)
        self.dropout = dropout
        '''

        self._output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = normalization.Normalizer(size=9, name='node_normalizer')
        self._world_edge_normalizer = normalization.Normalizer(size=4, name='world_edge_normalizer')
        
        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(nfeat, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid)
            ), 
            edge_dim=edge_dim  # Pass edge_dim to the GINEConv
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid)
            ), 
            edge_dim=edge_dim
        )
        self.conv3 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid)
            ), 
            edge_dim=edge_dim
        )
        self.conv4 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid)
            ), 
            edge_dim=edge_dim
        )
        self.conv5 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid)
            ), 
            edge_dim=edge_dim
        )
        self.conv6 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid)
            ), 
            edge_dim=edge_dim
        )
        self.dropout = dropout
        self.lin1 = nn.Linear(nhid, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, output)
        

    def forward(self, input_frame,is_training=True):

        x, edge_index, edge_attr, _ = build_graph(self._output_normalizer,self._world_edge_normalizer,input_frame)
        assert edge_index.dtype == torch.long
        assert edge_attr.dtype == torch.float
        '''
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, edge_index)
        x = F.relu(x) 
        x = self.out(x)
        '''
        print(f"Node feature size: {x.shape}")  # Expect shape: [num_nodes, feature_dim]
        print(f"Edge feature size: {edge_attr.shape}")  # Expect shape: [num_edges, feature_dim]

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(self.conv4(x, edge_index, edge_attr))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(self.conv5(x, edge_index, edge_attr))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(self.conv6(x, edge_index, edge_attr))
        
        # Fully connected layers for final predictions
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        
        if is_training:
            return x
        else:
            return self._update(input_frame, x)
        

    def _update(self, input_frame, x):
        velocity = self._output_normalizer.inverse(x[:, :3])
        current_position = input_frame['curr_pos']
        position = current_position + velocity
        return (position, current_position, velocity)
    
    def get_output_normalizer(self):
        return (self._output_normalizer)
