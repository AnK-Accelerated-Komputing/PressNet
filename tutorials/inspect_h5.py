"""Print the structure of a PressNet HDF5 file and optionally plot one trajectory.

    python tutorials/inspect_h5.py datasets/data/coarse/coarse_1500_train_val.h5
    python tutorials/inspect_h5.py data.h5 --group group_0 --step 1000 --plot
"""
import argparse
import json
import os

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5_path")
    parser.add_argument("--group", default=None, help="Trajectory to inspect (default: first)")
    parser.add_argument("--step", type=int, default=1500, help="Step to plot")
    parser.add_argument("--plot", action="store_true", help="3D scatter of the blank at --step, coloured by stress")
    args = parser.parse_args()

    meta_path = args.h5_path.replace(".h5", "_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    with h5py.File(args.h5_path, "r") as f:
        groups = list(f.keys())
        print(f"{len(groups)} trajectories")
        name = args.group or groups[0]
        g = f[name]
        print(f"\n{name}: {meta.get(name, {})}")
        for key in ("cells", "mesh_pos", "node_type"):
            print(f"  {key:<10} {g[key].shape} {g[key].dtype}")
        steps = [k for k in g.keys() if k.startswith("step_")]
        print(f"  {len(steps)} steps; step_1: " + ", ".join(f"{k} {g['step_1'][k].shape}" for k in g["step_1"]))
        types, counts = np.unique(g["node_type"][:], return_counts=True)
        print("  node types:", dict(zip(types.astype(int).tolist(), counts.tolist())))

        if args.plot:
            import matplotlib.pyplot as plt
            step = g[f"step_{args.step}"]
            pos, stress = step["world_pos"][:], step["stress"][:, 0]
            blank = g["node_type"][:, 0] == 0
            ax = plt.figure(figsize=(7, 5)).add_subplot(projection="3d")
            sc = ax.scatter(*pos[blank].T, c=stress[blank], s=3, cmap="viridis")
            plt.colorbar(sc, label="von Mises stress [MPa]")
            ax.set_title(f"{name}, step {args.step}")
            plt.show()


if __name__ == "__main__":
    main()
