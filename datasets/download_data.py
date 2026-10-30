"""Download the PressNet++ dataset files and arrange them as described in datasets/README.md.

The links are published together with the paper; until then the table below is empty and
the script only tells you which files it is waiting for.

    python datasets/download_data.py --mesh coarse
    python datasets/download_data.py --mesh fine --split train_val
"""
import argparse
import os
import urllib.request

# (mesh, split) -> direct-download URL of the .h5 ; the *_meta.json lives next to it under the same key + "_meta"
# Fill in when the dataset is published (Google Drive direct links or Zenodo URLs).
URLS = {
    # ("coarse", "train_val"): "https://...",
    # ("coarse", "train_val_meta"): "https://...",
}
SPLITS = ["train_val", "extrapolation", "unseen"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", required=True, choices=["coarse", "medium", "fine"])
    parser.add_argument("--split", default="all", choices=SPLITS + ["all"])
    parser.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else [args.split]
    out_dir = os.path.join(args.root, args.mesh)
    os.makedirs(out_dir, exist_ok=True)

    for split in splits:
        for suffix in ("", "_meta"):
            key = (args.mesh, split + suffix)
            name = f"{args.mesh}_1500_{split}" + (".h5" if not suffix else "_meta.json")
            target = os.path.join(out_dir, name)
            if os.path.exists(target):
                print(f"[have] {target}")
                continue
            if key not in URLS:
                print(f"[todo] {name}: download link not published yet")
                continue
            print(f"[get ] {name}")
            urllib.request.urlretrieve(URLS[key], target)


if __name__ == "__main__":
    main()
