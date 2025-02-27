import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer
from utilities import common

device = torch.device('cuda')

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
    # node_type = node_type.unsqueeze(0)  # Shape: (1, num_points)
    # print(x.size(1))
    return x

class regpointnet_seg(nn.Module):
    def __init__(self, output_size=3, input_channel=12):
        super(regpointnet_seg, self).__init__()
    
        self.output_size = output_size
        self.stn = STN3d(input_channel)
        self.conv1 = torch.nn.Conv1d(input_channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

        # InstanceNorm1d → Handles batch size = 1
        self.in1 = nn.InstanceNorm1d(64)
        self.in2 = nn.InstanceNorm1d(128)
        self.in3 = nn.InstanceNorm1d(128)
        self.in4 = nn.InstanceNorm1d(512)
        self.in5 = nn.InstanceNorm1d(2048)

        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4928, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, output_size, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, inputs):
        point_cloud = process_inputs(inputs, device)
        # print(point_cloud.shape)
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        # point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud, feature = point_cloud[:, :, :3], point_cloud[:, :, 3:]  # Now [B, N, 3] and [B, N, 9]

        point_cloud = torch.bmm(point_cloud, trans)
        point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        # out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        # out2 = F.relu(self.bn2(self.conv2(out1)))
        # out3 = F.relu(self.bn3(self.conv3(out2)))
        # Use InstanceNorm when B=1, otherwise use BatchNorm
        norm1 = self.in1 if B == 1 else self.bn1
        norm2 = self.in2 if B == 1 else self.bn2
        norm3 = self.in3 if B == 1 else self.bn3
        norm4 = self.in4 if B == 1 else self.bn4
        norm5 = self.in5 if B == 1 else self.bn5

        out1 = F.relu(norm1(self.conv1(point_cloud)))
        out2 = F.relu(norm2(self.conv2(out1)))
        out3 = F.relu(norm3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        # out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        # out5 = self.bn5(self.conv5(out4))
        out4 = F.relu(norm4(self.conv4(net_transformed)))
        out5 = norm5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        # out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.output_size), dim=-1)
        net = net.view(B, N, self.output_size) # [B, N, 3]

        if net.size(0) == 1:
            diff = net.squeeze(0)
        return diff


