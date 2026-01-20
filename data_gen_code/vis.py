import plotly.graph_objects as go
import numpy as np
import h5py

mesh_np = []
node_np = []

with h5py.File(r"J:\Files\03_Work\10_AnK\3_PressNetData\Data_From_Notion\1500_Medium\h5_medium\pfull_1500_medium.h5", 'r') as hfile:
    
    h5_groups = hfile.keys()

    for group in h5_groups:
        unit = hfile[group]
        
        mesh_pos_content = unit['mesh_pos']
        node_type_content = unit['node_type']

        mesh_np = np.array(mesh_pos_content, dtype=np.float32)
        node_np = np.array(node_type_content, dtype=np.float32)

        print(mesh_np)
        print(node_np)
        break

color_map = {
    3: 'blue',
    1: 'green',
    0: 'red'  # Assuming red for type 3
}

# Assign color to each node based on its type
colors = [color_map.get(int(val), 'gray') for val in node_np]

# Create a 3D scatter plot
fig = go.Figure(data=[
    go.Scatter3d(
        x=mesh_np[:, 0],
        y=mesh_np[:, 1],
        z=mesh_np[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            opacity=0.8
        )
    )
])

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    title='3D Visualization of Mesh Points Colored by Node Type'
)

# Show the plot
fig.show()
