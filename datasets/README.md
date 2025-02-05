# Details of PressNet Dataset

We have provided both the raw data extracted from ANSYS simulation as well as the processed and compressed .h5 file and corresponding meta json file.

For convience, we suggest directly using the data stored in .h5 file, while one can also inspect the raw files and use similar techniques to obtain personal custom training data from ANSYS.


## .h5 data file
This is the compressed binary file containing the information of all 150 trajectories in dictionary forma. So, there are 150 key-value pairs of keys "group_0", "group_1" ..... "group_149" and each value being of following format:
```json
"group_0": {
        "cells": {
            "type": "dataset",
            "shape": [952,4],
            "dtype": "float32"
        },
        "mesh_pos": {
            "type": "dataset",
            "shape": [442,3],
            "dtype": "float32"
        },
        "node_type": {
            "type": "dataset",
            "shape": [442,1],
            "dtype": "float32"
        },
        "step_1": {
            "stress": {
                "type": "dataset",
                "shape": [442,1],
                "dtype": "float32"
            },
            "curr_pos": {
                "type": "dataset",
                "shape": [442,3],
                "dtype": "float32"
            }
        },
        .
        .
        .
        .
        "step_n": {
            "stress": {
                "type": "dataset",
                "shape": [442,1],
                "dtype": "float32"
            },
            "curr_pos": {
                "type": "dataset",
                "shape": [442,3],
                "dtype": "float32"
            }
        }
}
```
### Description
- Each ```trajectory/group``` represent a single structural simulation. 
- ```mesh_pos``` represents the x, y and, z coordinate values of the mesh nodes at initial position or in the meshing
- ```node_type``` represents one of the three types of node: normal, handle or, obstacle
  - Normal node types are node lying in the deforming plate and undergoes deformation.
  - Handle node types are node lying in the deforming plate but in contact with the lower die in the initial state such that the vertical deformation is zero at the given point.
  - Obstacle node types are node lying in the upper die and lowe die body.
- ```cells``` represent the list of mesh elements (tetrahedral in our case). Each row of cell contains list of four node which compose the given cell. The nodes are represented based on their serial number.
- ```step_n```: The number of steps range from 1 to 1500 (or 1 to 400 in case of demo data) representing each step of the demo data.
  - Each step contain values of current node position and stress value
  - ```curr_pos``` corresponds the x, y and, z values of each node respectively at the given time step.
  - ```stress``` corresponds the von-misses-stress at each node respectively at the given time step.

## Meta JSON file
This file contains the information regarding the each corresponding trajectory/group in the .h5 file. The information includes:
- Die Shape
- Simulation Name
- Number of Nodes
- Number of Cells
- Del Time/ Time Step
- Number of Steps
- Down Motion Steps
- Stop Motion Steps
- Up Motion Steps


## Raw Files
There are 15 folders for each die shapes and each of such folder contains 10 different folders with simulation varying in geometric parameters of the given die shape. Such individual simulation folder contains two types of file, .dat file and .txt files

- .dat files is the ANSYS simulation input file and contain information regarding Mesh nodes, bodies, cells and contact bodies

- There are four folders with .txt files for x_deformation, y_deformation, z_deformation and stress for each step where each file contain the correspoding nodal values for all the nodes in the given simulation

```
15x10_400steps_coarse_data/
└── DATA_Channel_Rectangular/
    ├── rect_10x1_mm_plate_solid_185_data/
    │   │──── Directional_Deformation_X/
    |   |     |────Directional Deformation X 2.txt
    |   |     |────Directional Deformation X 3.txt
    |   |     |────............
    │   │──── Directional_Deformation_Y/
    │   │──── Directional_Deformation_Z/
    │   │──── Equivalent_Stress/
    │   │──── rect_10x1_mm_plate_solid_185.dat
    |
    └── rect_10x2_mm_plate_solid_185_data
    .......
    .......
└──DATA_CHANNEL_SEMI_CIRCULAR
................
................
```


## Download Guidelines
All the data are uploaded in this [Google Drive Link](https://drive.google.com/drive/folders/1i8qMkiWVyD5RynaXxqD_36vDmukQktFi?usp=sharing). One may directly download from the drive. In case of servers, it will be more convient to run the ```download_raw_data.py``` file and just changing the link to the respective RAR file. The current link in the python script downloads the ```15x10_400steps_coarse_data```.