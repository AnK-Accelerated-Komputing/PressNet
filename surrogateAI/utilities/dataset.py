import torch
from torch.utils.data import Dataset
import json
import h5py
import os
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None, stage=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.stage = stage
        self.is_processed = self._is_processed()

        if not self.is_processed:
            self.process_data()
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _is_processed(self):
        return os.path.exists(self.data_path.replace('.h5', f'_{self.split}_{self.stage}.pt'))

    def process_data(self):
        with open('meta.json', 'r') as f:
            meta_data = json.load(f)

        final_data = []
        with h5py.File(self.data_path, 'r') as h5_file:
            group_names = list(h5_file.keys())
            total_groups = len(group_names)
            
            # 70% for train, 15% val, 15% test
            if self.split == 'train':
                group_limit = round(total_groups * 0.7)
                selected_groups = group_names[:group_limit]
            elif self.split == 'val':
                group_limit_start = round(total_groups * 0.7)
                group_limit_end = round(total_groups * 0.85)
                selected_groups = group_names[group_limit_start:group_limit_end]
            else:  # test
                group_limit_start = round(total_groups * 0.85)
                selected_groups = group_names[group_limit_start:]

            for group_name in selected_groups:
                group = h5_file[group_name]
                meta_info = meta_data[group_name]

                total_steps = meta_info['steps']
                num_nodes = meta_info['number of nodes']
                num_cells = meta_info['number of cells']

                # Determine step range based on stage
                if self.stage == 1:
                    start_step = 1
                    end_step = total_steps // 3
                elif self.stage == 2:
                    start_step = total_steps // 3 + 1
                    end_step = 2 * (total_steps // 3)
                elif self.stage == 3:
                    start_step = 2 * (total_steps // 3) + 1
                    end_step = total_steps
                else:
                    start_step = 1
                    end_step = total_steps

                step_limit = end_step - start_step + 1

                # Common datasets
                cells = torch.tensor(group['cells'][:])  # Convert to tensor
                mesh_pos = torch.tensor(group['mesh_pos'][:])
                node_type = torch.tensor(group['node_type'][:])

                if self.split == 'train':
                    for step_num in range(start_step, end_step + 1):
                        step_key = f'step_{step_num}'
                        next_step_key = f'step_{step_num + 1}'

                        if step_key in group:
                            step_group = group[step_key]
                            curr_pos = torch.tensor(step_group['world_pos'][:])  # Convert to tensor

                            if next_step_key in group:
                                next_step_group = group[next_step_key]
                                next_pos = torch.tensor(next_step_group['world_pos'][:])
                            else:
                                next_pos = None

                            step_data = {
                                'cells': cells,
                                'mesh_pos': mesh_pos,
                                'node_type': node_type,
                                'curr_pos': curr_pos,
                                'next_pos': next_pos
                            }
                            final_data.append(step_data)
                else:
                    # For val/test, group the data as tensors
                    grouped_data = {
                        'cells': cells.repeat(step_limit, 1, 1),  # (steps, num_cells, 3)
                        'mesh_pos': mesh_pos.repeat(step_limit, 1, 1),  # (steps, num_nodes, 3)
                        'node_type': node_type.repeat(step_limit, 1),  # (steps, num_nodes)
                        'curr_pos': torch.zeros((step_limit, num_nodes, 3)),
                        'next_pos': torch.zeros((step_limit, num_nodes, 3))
                    }

                    for idx, step_num in enumerate(range(start_step, end_step + 1)):
                        step_key = f'step_{step_num}'
                        next_step_key = f'step_{step_num + 1}'

                        if step_key in group:
                            step_group = group[step_key]
                            grouped_data['curr_pos'][idx] = torch.tensor(step_group['world_pos'][:])

                        if next_step_key in group:
                            next_step_group = group[next_step_key]
                            grouped_data['next_pos'][idx] = torch.tensor(next_step_group['world_pos'][:])

                    final_data.append(grouped_data)

        # Save data to a split-specific .pt file
        torch.save(final_data, self.data_path.replace('.h5', f'_{self.split}_{self.stage}.pt'))
        print(f"Data successfully saved to {self.split}_{self.stage}.pt")

    def load_data(self):
        return torch.load(self.data_path.replace('.h5', f'_{self.split}_{self.stage}.pt'))
