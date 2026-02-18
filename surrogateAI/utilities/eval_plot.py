import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_evaluation_loss(exp_dir, epoch):
   
    # Build paths
    log_dir = os.path.join(exp_dir, 'log')
    eval_pkl = os.path.join(log_dir, f'eval_loss_epoch_{epoch}.pkl')
    if not os.path.isfile(eval_pkl):
        raise FileNotFoundError(f"Evaluation pickle not found: {eval_pkl}")

    # Load data
    df = pd.read_pickle(eval_pkl)
    # print(df)
    eval_losses = df['eval_mse_losses_pos']
    # print(len(eval_losses))
    epochs = list(range(1, len(eval_losses) * 20, 20))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs[0:], eval_losses[0:], marker='o', linestyle='-', linewidth=1.5, markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Evaluation Loss')
    plt.title(f'Evaluation Loss (epoch {epoch})')
    plt.grid(True)

    # Save
    save_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'eval_loss_epoch_{epoch}.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved evaluation plot to {out_path}")

def main():
    exp_dir = '/home/sushil/PressNet/datasetsoutput/encode_process_decode/coarse_1500_train_val/Sat-Jan-31-17-24-55-2026'
    epoch = 500
    plot_evaluation_loss(exp_dir, epoch)
    
if __name__ == '__main__':
    main()