import os
from pathlib import Path
import pickle
import time
import datetime

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

# from model import DGCNN, MagNet
from utilities import press_eval, common
from utilities.trajectory_solid185_quarter import generate_trajectory
from utilities.evalutation_trajectory import generate_evaluation_trajectory


device = torch.device('cuda')


def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        data_frame[k] = torch.squeeze(v, 0)
    return data_frame

def pickle_save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def loss_fn(inputs, network_output, model):
    """L2 loss on position."""
    # build target acceleration
    world_pos = inputs['world_pos']
    target_world_pos = inputs['target|world_pos']
    # target_stress = inputs['target|stress']
    
    cur_position = world_pos
    target_position = target_world_pos
    target_velocity = target_position - cur_position

    node_type = inputs['node_type']

    # world_pos_normalizer, stress_normalizer = model.get_output_normalizer()
    world_pos_normalizer = model.get_output_normalizer()
    target_normalized = world_pos_normalizer(target_velocity)
    # target_normalized_stress = stress_normalizer(target_stress).to(device)
    loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
    # print("loss mask shape", loss_mask)
    pos_prediction = network_output[:,:3]
    # stress_prediction = network_output[:,3:]
    # print("pos prediction shape", pos_prediction.shape)
    # print(pos_prediction)
    # error = torch.sum((target_normalized - pos_prediction) ** 2, dim=1)
    # print("error shape", error)
    # loss = torch.mean(error[loss_mask])

    error = torch.sum((target_normalized - pos_prediction) ** 2, dim=1)
    # error += torch.sum((target_normalized_stress - stress_prediction) ** 2, dim=1)
    # loss = torch.mean(error)
    loss = torch.mean(error[loss_mask])
    return  loss