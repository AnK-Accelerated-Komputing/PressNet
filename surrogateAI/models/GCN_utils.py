import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utilities import common
from models import normalization

device = torch.device('cuda')

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


from scipy.spatial.distance import cdist
import numpy as np
import scipy.sparse as sp
import torch
import collections

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])

def build_graph(trajectory, k=20):
    output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')
    node_normalizer = normalization.Normalizer(size=9, name='node_normalizer')
    world_edge_normalizer = normalization.Normalizer(size=4, name='world_edge_normalizer')
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



def normalize(mx):
    """Row-normalize sparse matrix"""
    if isinstance(mx, torch.Tensor):
        mx = mx.cpu()
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def inverse_normalize(mx):
    """Inverse of row-normalization for sparse matrix"""
    if isinstance(mx, torch.Tensor):
        mx = mx.cpu()
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    
    # Avoid division by zero by checking rowsum
    # Multiply each row of mx by the original row sums
    r_mat = sp.diags(rowsum)
    mx = r_mat.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
