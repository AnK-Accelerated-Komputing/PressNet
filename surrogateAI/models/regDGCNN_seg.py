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
import torch.nn.functional as F
import os
import copy
import math
import numpy as np
import trimesh
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, JumpingKnowledge
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import BatchNorm

from utilities import common
device = torch.device('cuda')
# def knn(x, k):
#     """
#     Computes the k-nearest neighbors for each point in x.

#     Args:
#         x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
#         k (int): The number of nearest neighbors to find.

#     Returns:
#         torch.Tensor: Indices of the k-nearest neighbors for each point, shape (batch_size, num_points, k).
#     """
#     # Calculate pairwise distance, shape (batch_size, num_points, num_points)
#     inner = -2 * torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)

#     # Retrieve the indices of the k nearest neighbors
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]
#     return idx

# def knn_filtered(x, node_types, k):
#     """
#     Compute k-NN for all nodes, but only return neighbors for non-obstacle nodes.
#     Neighbors can be from any node type.

#     Args:
#         x (torch.Tensor): Input tensor of shape (batch_size, num_dims, num_points).
#         node_types (torch.Tensor): Node type tensor of shape (batch_size, num_points).
#         k (int): Number of nearest neighbors.

#     Returns:
#         torch.Tensor: Indices of k-nearest neighbors for valid nodes.
#     """
#     # batch_size, _, num_points = x.shape

#     # Identify valid nodes (exclude obstacles)
#     valid_mask = node_types != 1  # Shape (batch_size, num_points)
    
#     # Compute pairwise distance using ALL nodes (no filtering yet)
#     inner = -2 * torch.matmul(x.transpose(2, 1), x)  
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)

#     # Find k nearest neighbors for ALL nodes
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     print("idx shape:", idx.shape)  # Should be (batch_size, num_points, k)
#     print("valid_mask shape before squeeze:", valid_mask.shape)
#     print("valid_mask shape after squeeze:", valid_mask.squeeze(0).shape)

#     # Apply mask to only keep indices corresponding to valid nodes
#     valid_idx = idx[:, valid_mask.squeeze(), :]  # Select only valid nodes

#     return valid_idx, valid_mask.squeeze()




# def get_graph_feature(x, node_types, k=20, idx=None):
#     """
#     Constructs local graph features for each point by finding its k-nearest neighbors and
#     concatenating the relative position vectors.

#     Args:
#         x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
#         node_types (torch.Tensor): Node type tensor (batch_size, num_points).
#         k (int): The number of neighbors to consider for graph construction.
#         idx (torch.Tensor, optional): Precomputed k-nearest neighbor indices.

#     Returns:
#         torch.Tensor: The constructed graph features of shape (batch_size, 2*num_dims, num_points, k).
#     """
#     # batch_size = x.size(0)
#     # num_points = x.size(2)
#     print("x shape:", x.shape)
#     batch_size, num_dims, og_num_points = x.shape
    

#     if idx is None:
#         idx, plate_mask = knn_filtered(x, node_types, k)
#     num_points = idx.shape[1]
#     x = x.view(batch_size, -1, og_num_points)
#     print("idx shape:", idx.shape)  # Should be [batch_size, num_points, k]
#     print("valid_mask shape:", plate_mask.shape)
    

#     # Compute k-nearest neighbors if not provided
#     # if idx is None:
#     #     idx = knn(x, k=k)

#     # Gather only plate nodes
#     plate_indices = torch.nonzero(plate_mask, as_tuple=True)
#     x_plate = x[:, :, plate_mask]  # Shape (batch_size, num_dims, num_plate_points)

#     # Prepare indices for gathering
#     device = x.device
#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
#     idx = idx + idx_base
#     idx = idx.view(-1)

#     # _, num_dims, _ = x.size()
#     x = x.transpose(2, 1).contiguous()
#     print("x.shape before view:", x.shape)
#     # Gather neighbors for each point to construct local regions
#     feature = x.view(batch_size * og_num_points, -1)[idx, :]
#     print("feature.shape before view:", feature.shape)
#     print("Expected size:", batch_size * num_points * k * num_dims)
#     print("Actual size:", feature.numel())  # total number of elements

#     feature = feature.view(batch_size, num_points, k, num_dims)

#     # Expand x_plate to match neighbor dimensions
#     x_plate = x_plate.transpose(2, 1).contiguous().view(batch_size, x_plate.shape[2], 1, num_dims).repeat(1, 1, k, 1)

#     # Expand x to match the dimensions for broadcasting subtraction
#     # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

#     # Concatenate the original point features with the relative positions to form the graph features
#     # feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
#     feature = torch.cat((feature - x_plate, x_plate), dim=3).permute(0, 3, 1, 2).contiguous()

#     return feature

def intermediate(x, xx):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    torch.cuda.empty_cache()
    return -xx - inner

def knn(x, k):
    x = x.to(torch.float16)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = intermediate(x, xx) - xx.transpose(2, 1)
    torch.cuda.empty_cache()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    torch.cuda.empty_cache()
    return feature      # (batch_size, 2*num_dims, num_points, k)

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
    world_pos = inputs['curr_pos'].to(device)  # Shape: (num_points, num_dims)
    node_type = inputs['node_type'].to(device)  # Shape: (num_points,)

    # Convert node_type to one-hot encoding
    one_hot_node_type = nn.functional.one_hot(node_type.to(torch.int64), common.NodeType.SIZE).float()  # Shape: (num_points, num_node_types)
    # print(world_pos.shape, one_hot_node_type.shape)
    one_hot_node_type = one_hot_node_type.squeeze(1)  # Removes the extra dim


    # Concatenate world position with one-hot encoded node types
    x = torch.cat((world_pos, one_hot_node_type), dim=1)  # Shape: (num_points, num_features)

    # Add batch dimension (batch_size = 1) and permute
    x = x.T.unsqueeze(0)  # Shape: (1, num_features, num_points)

    # Reshape node_type to include batch dimension
    node_type = node_type.unsqueeze(0)  # Shape: (1, num_points)
    # print(x.size(1))
    return x, node_type

class regDGCNN_seg(nn.Module):
    def __init__(self, output_size=3, input_dims=12, k=20, emb_dims=1024, dropout=0.5):
        super(regDGCNN_seg, self).__init__()
        self.output_size = output_size
        self.input_dims = input_dims
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_dims*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.dropout)
        self.conv9 = nn.Conv1d(256, output_size, kernel_size=1, bias=False)


    def forward(self, inputs):
        x,node_types = process_inputs(inputs,device)
        batch_size = x.size(0)
        num_points = x.size(2)
        # print("input dimes is",x.size(1))

        x = get_graph_feature(x, k=self.k)   # (batch_size, input_dims, num_points) -> (batch_size, input_dims*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, input_dims*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)      # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, output_size, num_points)

        if x.size(0) == 1:
            x = x.squeeze(0)
            x = x.permute(1, 0)
        return x

