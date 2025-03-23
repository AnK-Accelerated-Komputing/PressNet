import pandas as pd
import matplotlib.pyplot as plt
import os

train_pkl_path = '/home/user/PressNet/surrogateAI/training_output/regDGCNN_seg/Channel_rect_press_dataset/Tue-Mar-18-13-42-20-2025/log/temp_train_loss_500.pkl'
train_df = pd.read_pickle(train_pkl_path)
train_epoch_losses = [loss.item() for loss in train_df['train_epoch_losses']]
# Plotting
epochs = range(2, len(train_epoch_losses)+1 )  # Epochs start from 1
plot_path = train_pkl_path.replace(train_pkl_path.split("/")[-1],"plots/training_loss.png")
print("creating plot, will save in",plot_path)
os.makedirs(os.path.dirname(plot_path),exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_epoch_losses[1:], marker='o', color='b', linestyle='-', linewidth=1.5, markersize=5)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs. Epoch")
plt.grid(True)
plt.savefig(plot_path)
print(f"Training loss plot saved to {plot_path} ")
# eval_df = pd.read_pickle(eval_pkl_path)