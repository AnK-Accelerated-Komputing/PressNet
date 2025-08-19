#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

This module is used to define both point-cloud based and graph-based models, including RegDGCNN, PointNet, and several Graph Neural Network (GNN) models
for the task of surrogate modeling of the aerodynamic drag.
"""
import torch
import torch.nn as nn

from utilities import common

device = torch.device('cuda')

def intermediate(x, xx):
    """
    Compute the intermediate matrix used in pairwise squared-distance computation.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, N) containing point features.
        xx (torch.Tensor): Precomputed squared-norm tensor of shape (B, 1, N).

    Returns:
        torch.Tensor: Tensor of shape (B, N, N) containing the intermediate term (-xx - (-2 * x^T x))
                      used as part of the pairwise squared-distance calculation.
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    torch.cuda.empty_cache()
    return -xx - inner

def knn(x, k):
    """
    Compute k-nearest neighbor indices for each point using pairwise squared distances.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, N) containing point features.
        k (int): Number of nearest neighbors to return per point.

    Returns:
        torch.LongTensor: Indices tensor of shape (B, N, k) containing the indices of the k nearest
                          neighbors for each of the N points in each batch.
    """
    x = x.to(torch.float16)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = intermediate(x, xx) - xx.transpose(2, 1)
    torch.cuda.empty_cache()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    """
    Build local edge features for graph-based convolutions by gathering k-nearest neighbor features.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, N) (batch_size, num_dims, num_points).
        k (int, optional): Number of neighbors to use. Default is 20.
        idx (torch.LongTensor, optional): Precomputed neighbor indices of shape (B, N, k).
                                          If provided, knn will not be computed.
        dim9 (bool, optional): If True, compute neighbor indices using a subset of channels
                               (channels starting from index 6) instead of the full feature set.

    Returns:
        torch.Tensor: Edge feature tensor of shape (B, 2 * C, N, k), where features consist of
                      the concatenation of (neighbor - center) and center for each point and its k neighbors.
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k) 
        else:
            idx = knn(x[:, 6:], k=k)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous() 
    feature = x.view(batch_size*num_points, -1)[idx, :]
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

class regDGCNN_seg(nn.Module):
    def __init__(self, output_size=3, input_dims=12, k=20, emb_dims=1024, dropout=0.5):
        """
        Initialize the regDGCNN_seg model architecture and layers.

        Args:
            output_size (int, optional): Number of output channels per point. Default 3.
            input_dims (int, optional): Number of input feature channels per point. Default 12.
            k (int, optional): Number of nearest neighbors used by graph feature construction. Default 20.
            emb_dims (int, optional): Dimension of the global embedding produced before classification/regression. Default 1024.
            dropout (float, optional): Dropout probability applied before the final layer. Default 0.5.

        Returns:
            None
        """
        super(regDGCNN_seg, self).__init__()
        self.output_size = output_size
        self.input_dims = input_dims
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout
        self.gn1 = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.gn2 = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.gn3 = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.gn4 = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.gn5 = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.gn6 = nn.GroupNorm(num_groups=8,  num_channels=self.emb_dims)
        self.gn7 = nn.GroupNorm(num_groups=8,  num_channels=512)
        self.gn8 = nn.GroupNorm(num_groups=8,  num_channels=256)
        self.conv1 = nn.Sequential(nn.Conv2d(self.input_dims*2, 64, kernel_size=1, bias=False),
                                   self.gn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.gn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.gn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.gn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.gn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.gn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.gn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.gn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.dropout)
        self.conv9 = nn.Conv1d(256, output_size, kernel_size=1, bias=False)

    def forward(self, inputs):
        """
        Forward pass: process inputs, build graph features, apply sequential graph convolutions, and produce per-point outputs.

        Args:
            inputs (dict): Dictionary expected to contain:
                - 'curr_pos' (torch.Tensor): Current world positions tensor used to build input features.
                - 'node_type' (torch.Tensor): Node type tensor used to build one-hot encodings.
            The tensors are moved to the device inside process_inputs.

        Returns:
            torch.Tensor: If batch size == 1, returns a tensor of shape (num_points, output_size).
                        Otherwise returns a tensor of shape (batch_size, output_size, num_points).
        """
        x,node_types = process_inputs(inputs,device)
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k)   
        x = self.conv1(x)                      
        x = self.conv2(x)                      
        x1 = x.max(dim=-1, keepdim=False)[0]   

        x = get_graph_feature(x1, k=self.k)     
        x = self.conv3(x)                      
        x = self.conv4(x)                       
        x2 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x2, k=self.k)     
        x = self.conv5(x)                       
        x3 = x.max(dim=-1, keepdim=False)[0]    

        x = torch.cat((x1, x2, x3), dim=1)     
        x = self.conv6(x)                       
        x = x.max(dim=-1, keepdim=True)[0]      
        x = x.repeat(1, 1, num_points)          
        x = torch.cat((x, x1, x2, x3), dim=1)   

        x = self.conv7(x)                       
        x = self.conv8(x)                       
        x = self.dp1(x)
        x = self.conv9(x)                       

        if x.size(0) == 1:
            x = x.squeeze(0)
            x = x.permute(1, 0)
        return x

