"""Functions to build evaluation metrics for cloth data."""

import torch
from utilities import common
import numpy as np
import mpl_toolkits.mplot3d as p3d

device = torch.device('cuda')


def _rollout(model, initial_state, num_steps, target_world_pos):
    """Rolls out a model trajectory."""
    node_type = initial_state['node_type'].to(device)
    mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
    mask = torch.stack((mask, mask, mask), dim=1)

    obstacle_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
    obstacle_mask = torch.stack((obstacle_mask, obstacle_mask, obstacle_mask), dim=1)

    def step_fn(cur_pos, trajectory, cur_positions, cur_velocities, target_world_pos):
        # memory_prev = torch.cuda.memory_allocated(device) / (1024 * 1024)
        with torch.no_grad():
            prediction, cur_position, cur_velocity = model({**initial_state, 'curr_pos': cur_pos, 'next_pos': target_world_pos}, is_training=False)

        next_pos = torch.where(mask, prediction, target_world_pos)
        # next_pos = prediction
        # next_pos = torch.where(obstacle_mask, torch.squeeze(target_world_pos), next_pos)

        trajectory.append(next_pos)
        cur_positions.append(cur_position)
        cur_velocities.append(cur_velocity)
        return next_pos, trajectory, cur_positions, cur_velocities

    cur_pos = torch.squeeze(initial_state['curr_pos'].to(device), 0)
    trajectory = []
    cur_positions = []
    cur_velocities = []
    for step in range(num_steps):
        cur_pos, trajectory, cur_positions, cur_velocities = step_fn(cur_pos, trajectory, cur_positions, cur_velocities, target_world_pos[step])
    return (torch.stack(trajectory), torch.stack(cur_positions), torch.stack(cur_velocities))

def evaluate(model, trajectory, num_steps=None):
    """Performs model rollouts and create stats."""
    initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}
    # print("??????INITIAL STATE?????????")
    # print(initial_state['mesh_pos'].shape)
    # print(initial_state['node_type'].shape)
    # print("meshpos",trajectory['mesh_pos'].shape)
    if num_steps is None:
        num_steps = trajectory['mesh_pos'].squeeze(0).shape[0]
    prediction, cur_positions, cur_velocities = _rollout(model, initial_state, num_steps, trajectory['next_pos'].squeeze(0).to(device))

    # error = tf.reduce_mean((prediction - trajectory['world_pos'])**2, axis=-1)
    # scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
    #            for horizon in [1, 10, 20, 50, 100, 200]}

    scalars = None

    # temp solution for visualization

    faces = trajectory['cells'].squeeze(0)
    # print(faces.shape)
    faces_result = []
    # print(faces.shape)
    for faces_step in faces:
        later = torch.cat((faces_step[:, 2:4], torch.unsqueeze(faces_step[:, 0], 1)), -1)
        faces_step = torch.cat((faces_step[:, 0:3], later), 0)
        faces_result.append(faces_step)
        # print(faces_step.shape)
    faces_result = torch.stack(faces_result, 0)
    # print(faces_result.shape)
    # print(faces_result[100].shape)


    # trajectory_polygons = to_polygons(trajectory['cells'], trajectory['curr_pos'])

    traj_ops = {
        # 'faces': trajectory['cells'],
        'faces': faces_result,
        'mesh_pos': trajectory['mesh_pos'],
        # 'gt_pos': trajectory_polygons,
        'gt_pos': trajectory['curr_pos'],
        'pred_pos': prediction,
        'cur_positions': cur_positions,
        'cur_velocities': cur_velocities,
        'node_type': trajectory['node_type']
    }
    return scalars, traj_ops
