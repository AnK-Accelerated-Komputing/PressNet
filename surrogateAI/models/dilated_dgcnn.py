import numpy as np
import torch
from torch import nn
from typing import Tuple
from utilities import common
try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps
except Exception as e:
    print("Warning: Cannot import furthest point sample. Using a slower version. Did install the PointNet++ Ops Lib? (See README.md)")
    def fps(xyz, npoint):
        """
        Farthest Point Sampling (FPS) algorithm for selecting a subset of points from a point cloud.

        Args:
            xyz (torch.Tensor): Input point cloud tensor of shape (B, N, C), where B is the batch size, N is the number of points, and C is the number of dimensions.
            npoint (int): Number of points to select.

        Returns:
            torch.Tensor: Tensor of shape (B, npoint) containing the indices of the selected points.
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

def knn(x, k=16):
    """
    Performs k-nearest neighbors (knn) search on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_points, num_dims).
        k (int): Number of nearest neighbors to find.

    Returns:
        torch.Tensor: Index tensor of shape (batch_size, num_points, k), containing the indices of the k nearest neighbors for each point.
    """
    x_t = x.transpose(2, 1)
    pairwise_distance = torch.cdist(x_t, x_t, p=2)
    idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def get_graph_feature(x, k=20, idx=None, pos=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if pos is None:
            idx = knn(x, k=k)
        else:
            idx = knn(pos, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    torch.cuda.empty_cache()

    return feature

def process_inputs(inputs, device):
    """
    Processes input tensors and prepares features for graph construction.

    Args:
        inputs (dict): A dictionary containing 'curr_pos' and 'node_type'.
        device (torch.device): The device to move tensors to.

    Returns:
        torch.Tensor: The processed feature tensor (1, num_features, num_points).
        torch.Tensor: Node type tensor for filtering.
    """
    world_pos = inputs['curr_pos'].to(device)  
    node_type = inputs['node_type'].to(device)  

    # Convert node_type to one-hot encoding
    one_hot_node_type = nn.functional.one_hot(node_type.to(torch.int64), common.NodeType.SIZE).float()  
    one_hot_node_type = one_hot_node_type.squeeze(1) 

    # Concatenate world position with one-hot encoded node types
    x = torch.cat((world_pos, one_hot_node_type), dim=1) 

    # Add batch dimension (batch_size = 1) and permute
    x = x.T.unsqueeze(0)  

    # Reshape node_type to include batch dimension
    node_type = node_type.unsqueeze(0) 
    return x, node_type

class dilated_dgcnn(nn.Module):
    """
    Graph-based segmentation network using Dynamic Graph CNN (DGCNN) blocks.
    Processes point cloud inputs and produces per-point predictions.

    Args:
        output_size (int): Number of output channels per point (e.g., classes or regression dims).
        input_dims (int): Dimensionality of input point features.
        k (int): Number of nearest neighbors for graph construction.
        emb_dims (int): Embedding dimensionality in the bottleneck layer.
        dropout (float): Dropout probability for final classifier.
    """

    def __init__(
        self,
        output_size: int = 3,
        input_dims: int = 6,
        k: int = 20,
        dilated_k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5
    ):
        super().__init__()
        self.output_size = output_size
        self.input_dims = input_dims
        self.k = k
        self.dilated_k = dilated_k
        self.emb_dims = emb_dims
        self.dropout = dropout

        # GroupNorm layers for 2D convolutions (graph feature extractors)
        self.gn1a  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.gn1b  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.gn2a  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.gn2b  = nn.GroupNorm(num_groups=8,  num_channels=64)

        # GroupNorm layers for 1D convolutions 
        self.gn5   = nn.GroupNorm(num_groups=16, num_channels=64)  
        
        self.gn6   = nn.GroupNorm(num_groups=32, num_channels=1024)
        self.gn7  = nn.GroupNorm(num_groups=16, num_channels=512)
        self.gn8   = nn.GroupNorm(num_groups=16, num_channels=256)
        
        # Graph feature extraction layers
        self.conv1a = nn.Sequential(
            nn.Conv2d(input_dims * 2, 64, kernel_size=1, bias=False),
            self.gn1a,
            nn.LeakyReLU(0.2)
        )  # First-edge features
        self.conv1b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.gn1b,
            nn.LeakyReLU(0.2)
        )  # Second-edge features

        # Second graph block uses concatenated features
        self.conv2a = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.gn2a,
            nn.LeakyReLU(0.2)
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.gn2b,
            nn.LeakyReLU(0.2)
        )
        
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.gn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # Final per-point classifier
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                        self.gn6,
                                        nn.LeakyReLU(negative_slope=0.2))
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1, bias=False),
            self.gn7,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            self.gn8,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dp1 = nn.Dropout(self.dropout)
        self.conv9 = nn.Conv1d(256, output_size, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            inputs: Raw point cloud tensor (e.g., [B, dims, N]).
            device: Torch device for processing.

        Returns:
            Per-point predictions of shape [B, output_size, N] or [N, output_size] if B==1.
        """

        # Preprocess inputs (e.g., feature selection or encoding)

        inputs,node_types = process_inputs(inputs,torch.device('cuda'))
        B, C, N = inputs.size()
        pos = inputs[:,:3,:].transpose(2,1)
        cd = torch.cdist(pos , pos, p=2)

        # ---- Graph block 1 ----
        x1 = self._dilated_dgcnn_block(inputs, pos, dilation_k=self.dilated_k, k=self.k, cd=cd, convs=[self.conv1a, self.conv1b])
        # ---- Graph block 2 ----
        x2 = self._dilated_dgcnn_block(x1, pos, dilation_k=self.dilated_k, k=self.k, cd=cd, convs=[self.conv2a, self.conv2b])
        x3 = self._dilated_dgcnn_block(x2, pos, dilation_k=self.dilated_k, k=self.k, cd=cd, convs=[self.conv5])
  

        x = torch.cat((x1, x2, x3), dim=1)     
        x = self.conv6(x)                       
        x = x.max(dim=-1, keepdim=True)[0]      
        x = x.repeat(1, 1, N)          
        x = torch.cat((x, x1, x2, x3), dim=1)   

        x = self.conv7(x)                       
        x = self.conv8(x)                       
        x = self.dp1(x)
        x = self.conv9(x)      
        
        
        # If batch size is 1, return [N, output_size]
        if B == 1:
            x = x.squeeze(0).permute(1, 0)

        return x

    def _dgcnn_block(
        self,
        x: torch.Tensor,
        k: int,
        convs: Tuple[nn.Sequential, nn.Sequential]
    ) -> torch.Tensor:
        """
        Helper for two-layer DGCNN block: feature extraction, neighbor aggregation, and max pooling.
        """
        # Build graph features: [B, C*2, N, k]
        x = get_graph_feature(x, k=k)

        # Apply first conv + activation
        x = convs[0](x)

        # Apply second conv + activation
        x = convs[1](x)

        # Aggregate via max over neighbors
        return x.max(dim=-1, keepdim=False)[0]

    def _dilated_dgcnn_block(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        dilation_k : int,
        k: int,
        cd: torch.Tensor,
        convs: Tuple[nn.Sequential, nn.Sequential]
    ) -> torch.Tensor:
        """
        Helper for dilated DGCNN block: feature extraction with dilation, neighbor aggregation, and max pooling.
        """
        B, C, N = x.shape
        if cd is None:
            cd = torch.cdist(pos , pos, p=2)
        inds = torch.topk(cd, dilation_k, largest=False).indices
        idx_l = inds.reshape(B * N, -1)
        idx_fps = fps(pos.reshape(B * N, -1)[idx_l], k).long()
        idx_fps = batched_index_select(idx_l, 1, idx_fps).reshape(B, N, -1)
        x = get_graph_feature(x, k=k, idx=idx_fps)
        for conv in convs:
            x = conv(x)

        x = x.max(dim=-1, keepdim=False)[0]
        return x
    