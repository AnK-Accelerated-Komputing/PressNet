import torch
from torch.utils.data import Dataset
import json
import h5py
import os
import numpy as np
import random

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, split='train', stage=None):
        self.data_path = data_path
        self.split = split
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
        print("processing data...")
        base_folder = os.path.dirname(self.data_path)
        data_name = self.data_path.split('/')[-1].split('.')[0].replace('_press_dataset','')
        meta_file = os.path.join(base_folder, f'{data_name}_meta.json')
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)

        final_data = []
        with h5py.File(self.data_path, 'r') as h5_file:
            group_names = list(h5_file.keys())
            random.seed(42)   # fixed seed for reproducibility
            random.shuffle(group_names)
            total_groups = len(group_names)
            
            # 70% for train, 15% val, 15% test
            if self.split == 'train':
                group_limit = round(total_groups * 0.7)
                selected_groups = group_names[:group_limit]
            elif self.split == 'val':
                group_limit_start = round(total_groups * 0.7)
                group_limit_end = round(total_groups * 0.9)
                if group_limit_end == group_limit_start:
                    selected_groups = group_names[group_limit_start:]
                else:
                    selected_groups = group_names[group_limit_start:group_limit_end]
            else:  # test
                group_limit_start = round(total_groups * 0.9)
                if group_limit_start == total_groups:
                    group_limit_start -= 1
                selected_groups = group_names[group_limit_start:]
            print(f"selected groups: {selected_groups}")
            for group_name in selected_groups:
                print(f"processing {group_name}")
                group = h5_file[group_name]
                meta_info = meta_data[group_name]

                total_steps = meta_info['number of steps']
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
                    end_step = total_steps-2                # -2 is introduced so that end step is 399
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
                                next_stress = torch.tensor(next_step_group['stress'][:])
                            else:
                                next_pos = None

                            step_data = {
                                'cells': cells,
                                'mesh_pos': mesh_pos,
                                'node_type': node_type,
                                'curr_pos': curr_pos,
                                'next_pos': next_pos,
                                'next_stress': next_stress
                            }
                            final_data.append(step_data)
                else:
                    # For val/test, accumulate the data in lists
                    curr_pos_list = []
                    next_pos_list = []
                    next_stress_list = []

                    for step_num in range(start_step, end_step + 1):
                        step_key = f'step_{step_num}'
                        next_step_key = f'step_{step_num + 1}'

                        if step_key in group:
                            step_group = group[step_key]
                            curr_pos_list.append(torch.tensor(step_group['world_pos'][:]))

                        if next_step_key in group:
                            next_step_group = group[next_step_key]
                            next_pos_list.append(torch.tensor(next_step_group['world_pos'][:]))
                            next_stress_list.append(torch.tensor(next_step_group['stress'][:]))

                    # Convert lists to tensors
                    curr_pos_tensor = torch.stack(curr_pos_list) if curr_pos_list else torch.empty(0)
                    next_pos_tensor = torch.stack(next_pos_list) if next_pos_list else torch.empty(0)
                    next_stress_tensor = torch.stack(next_stress_list) if next_stress_list else torch.empty(0)

                    grouped_data = {
                        'cells': cells.expand(curr_pos_tensor.size(0), -1, -1),  # (steps, num_cells, 3)
                        'mesh_pos': mesh_pos.expand(curr_pos_tensor.size(0), -1, -1),  # (steps, num_nodes, 3)
                        'node_type': node_type.expand(curr_pos_tensor.size(0),-1, -1),  # (steps, num_nodes,1)
                        'curr_pos': curr_pos_tensor,
                        'next_pos': next_pos_tensor,
                        'next_stress': next_stress_tensor
                    }

                    final_data.append(grouped_data)

        # Save data to a split-specific .pt file
        torch.save(final_data, self.data_path.replace('.h5', f'_{self.split}_{self.stage}.pt'))
        print(f"Data successfully saved to {self.split}_{self.stage}.pt")

    def load_data(self):
        return torch.load(self.data_path.replace('.h5', f'_{self.split}_{self.stage}.pt'))


def main():
    data_file = "/home/gd_user1/AnK/project_PINN/PressNet/surrogateAI/data/press_dataset.h5"
    dataset = TrajectoryDataset(data_file, split='train', stage=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    i=0
    for data in dataloader:
        i+=1
        # print(i,data.keys())
        if i ==1:
            for keys in data.keys():
                print(keys,data[keys].shape)
    

    return


if __name__ == '__main__':
    main()