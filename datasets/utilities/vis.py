import plotly.graph_objects as go
import numpy as np
import h5py
import os

# Map node types to colors based on the PressNet logic you're using:
# 0: Normal (Plate deforming), 1: Obstacle (Die), 3: Handle (Plate non-deforming)
color_map = {
    0: 'blue',   # Normal nodes
    1: 'green',  # Obstacle (Die) nodes
    3: 'red',    # Handle nodes
}

def visualize_h5_groups(h5_file_path):
    print(f"Loading {h5_file_path} for visual check...")
    
    if not os.path.exists(h5_file_path):
        print(f"❌ Error: File not found at {h5_file_path}")
        return

    with h5py.File(h5_file_path, 'r') as hfile:
        groups = list(hfile.keys())
        print(f"Found {len(groups)} groups: {groups}")
        
        for idx, group_name in enumerate(groups):
            print(f"\n--- [{idx + 1}/{len(groups)}] Visualizing {group_name} ---")
            group_data = hfile[group_name]
            
            # Load base mesh positions and node types
            mesh_np = np.array(group_data['mesh_pos'], dtype=np.float32)
            node_np = np.array(group_data['node_type'], dtype=np.float32).flatten()

            # Assign colors safely (fallback to gray if an unknown type exists)
            colors = [color_map.get(int(val), 'gray') for val in node_np]

            # Create a 3D scatter plot
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=mesh_np[:, 0],
                    y=mesh_np[:, 1],
                    z=mesh_np[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colors,
                        opacity=0.8
                    )
                )
            ])

            # Update layout for proper scaling
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data' # Ensures true geometric proportions
                ),
                title=f'Verification: 3D Mesh of {group_name}'
            )

            # Show the plot in the browser/viewer
            fig.show()
            
            # Pause execution until the user is ready for the next one
            if idx < len(groups) - 1:
                user_input = input("Press Enter to visualize the next group (or type 'q' to quit): ")
                if user_input.lower().strip() == 'q':
                    print("Exiting visualization.")
                    break
            else:
                print("Finished checking all groups in the file.")

if __name__ == "__main__":
    # Change this to whichever file you want to inspect:
    # medium_1500_explicit.h5 OR medium_1500_ranges.h5
    h5_test_file = r"/home/sushil/PressNet/_Nas_Mount/ank-server1/prj_accelerated_physics/prj_pressnet/00_data/Splitdata/Coarse/coarse_1500_explicit.h5"
    
    visualize_h5_groups(h5_test_file)