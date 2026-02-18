import pickle
import torch

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pkl_path = "/home/sushil/PressNet/_NAS_MOUNT/ank-server1/prj_accelerated_physics/prj_pressnet/Training_data_Susil/Server6/output/fine/gcn/fine_1500_train_val/GCN_F_ST1_edge8/log/temp_train_loss.pkl"

loss_record = load_pickle(pkl_path)

# ---- Extract values ----
epoch_losses = loss_record["train_epoch_losses"]

epoch_losses_tensor = torch.stack(epoch_losses)

max_epoch_loss = torch.max(epoch_losses_tensor).item()
min_epoch_loss = torch.min(epoch_losses_tensor).item()
mean_epoch_loss = torch.mean(epoch_losses_tensor).item()

print("==== Epoch Loss Statistics ====")
print(f"Max Epoch Loss : {max_epoch_loss:.6f}")
print(f"Min Epoch Loss : {min_epoch_loss:.6f}")
print(f"Mean Epoch Loss: {mean_epoch_loss:.6f}")
