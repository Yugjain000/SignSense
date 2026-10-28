import os
import h5py
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = r"d:\Final Project\research_model"

print("=" * 60)
print("Converting Springer Keras H5 Weights to PyTorch .pt")
print("=" * 60)

def convert_lstm_weights(k_kernel, k_recurrent, k_bias):
    # Keras order: [i, c, f, o]
    units = k_kernel.shape[1] // 4
    Wi, Wc, Wf, Wo = np.split(k_kernel, 4, axis=1)
    Ui, Uc, Uf, Uo = np.split(k_recurrent, 4, axis=1)
    bi, bc, bf, bo = np.split(k_bias, 4)

    # PyTorch order: [i, f, g(c), o]
    w_ih = np.concatenate([Wi, Wf, Wc, Wo], axis=1).T
    w_hh = np.concatenate([Ui, Uf, Uc, Uo], axis=1).T
    b_ih = np.concatenate([bi, bf, bc, bo])
    b_hh = np.zeros_like(b_ih)
    return (
        torch.tensor(w_ih, dtype=torch.float32),
        torch.tensor(w_hh, dtype=torch.float32),
        torch.tensor(b_ih, dtype=torch.float32),
        torch.tensor(b_hh, dtype=torch.float32),
    )

# -------------------------------------------------------------
# 1. CNN + LSTM (17 Classes, 99.88% Accuracy)
# -------------------------------------------------------------
class CNN_LSTM_17(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(225, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm1 = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 17)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, 50, 225) -> (B, 225, 50)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        # x: (B, 64, 11) -> (B, 11, 64) for LSTM
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        out, _ = self.lstm3(x)
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return self.softmax(logits)

print("\n[1/2] Converting CNN+LSTM (17 Classes)...")
f17 = h5py.File(os.path.join(BASE_DIR, "CNN_LSTM_weights", "model.weights.h5"), "r")
model_17 = CNN_LSTM_17()

# Conv1
c1_w = f17[r"_layer_checkpoint_dependencies\conv1d"]["vars"]["0"][:] # (3, 225, 32)
c1_b = f17[r"_layer_checkpoint_dependencies\conv1d"]["vars"]["1"][:]
model_17.conv1.weight.data = torch.tensor(c1_w, dtype=torch.float32).permute(2, 1, 0)
model_17.conv1.bias.data = torch.tensor(c1_b, dtype=torch.float32)

# Conv2
c2_w = f17[r"_layer_checkpoint_dependencies\conv1d_1"]["vars"]["0"][:] # (3, 32, 64)
c2_b = f17[r"_layer_checkpoint_dependencies\conv1d_1"]["vars"]["1"][:]
model_17.conv2.weight.data = torch.tensor(c2_w, dtype=torch.float32).permute(2, 1, 0)
model_17.conv2.bias.data = torch.tensor(c2_b, dtype=torch.float32)

# LSTMs helper
def load_bilstm(lstm_module, f_grp, b_grp):
    fw_k = f_grp["vars"]["0"][:]
    fw_r = f_grp["vars"]["1"][:]
    fw_b = f_grp["vars"]["2"][:]
    w_ih_l0, w_hh_l0, b_ih_l0, b_hh_l0 = convert_lstm_weights(fw_k, fw_r, fw_b)
    lstm_module.weight_ih_l0.data = w_ih_l0
    lstm_module.weight_hh_l0.data = w_hh_l0
    lstm_module.bias_ih_l0.data = b_ih_l0
    lstm_module.bias_hh_l0.data = b_hh_l0

    bw_k = b_grp["vars"]["0"][:]
    bw_r = b_grp["vars"]["1"][:]
    bw_b = b_grp["vars"]["2"][:]
    w_ih_l0_rev, w_hh_l0_rev, b_ih_l0_rev, b_hh_l0_rev = convert_lstm_weights(bw_k, bw_r, bw_b)
    lstm_module.weight_ih_l0_reverse.data = w_ih_l0_rev
    lstm_module.weight_hh_l0_reverse.data = w_hh_l0_rev
    lstm_module.bias_ih_l0_reverse.data = b_ih_l0_rev
    lstm_module.bias_hh_l0_reverse.data = b_hh_l0_rev

load_bilstm(
    model_17.lstm1,
    f17[r"_layer_checkpoint_dependencies\bidirectional\forward_layer\cell"],
    f17[r"_layer_checkpoint_dependencies\bidirectional\backward_layer\cell"]
)
load_bilstm(
    model_17.lstm2,
    f17[r"_layer_checkpoint_dependencies\bidirectional_1\forward_layer\cell"],
    f17[r"_layer_checkpoint_dependencies\bidirectional_1\backward_layer\cell"]
)
load_bilstm(
    model_17.lstm3,
    f17[r"_layer_checkpoint_dependencies\bidirectional_2\forward_layer\cell"],
    f17[r"_layer_checkpoint_dependencies\bidirectional_2\backward_layer\cell"]
)

# Dense
fc_w = f17[r"_layer_checkpoint_dependencies\dense"]["vars"]["0"][:] # (128, 17)
fc_b = f17[r"_layer_checkpoint_dependencies\dense"]["vars"]["1"][:]
model_17.fc.weight.data = torch.tensor(fc_w, dtype=torch.float32).t()
model_17.fc.bias.data = torch.tensor(fc_b, dtype=torch.float32)

f17.close()

# Save PyTorch Model
pt_17_path = os.path.join(BASE_DIR, "cnn_lstm_17.pt")
torch.save(model_17.state_dict(), pt_17_path)
print(f"[+] CNN+LSTM successfully saved to {pt_17_path}!")

model_17.eval()
with torch.no_grad():
    dummy = torch.randn(1, 50, 225)
    out17 = model_17(dummy)
    print(f"[+] CNN+LSTM test inference output: shape={out17.shape}, sum={out17.sum().item():.4f}")


# -------------------------------------------------------------
# 2. Stacked LSTM (61 Classes, 98.5% Accuracy)
# -------------------------------------------------------------
class StackLSTM_61(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(225, 64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.lstm5 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 61)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        out, _ = self.lstm5(x)
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return self.softmax(logits)

print("\n[2/2] Converting StackLSTM (61 Classes)...")
f61 = h5py.File(os.path.join(BASE_DIR, "weights_61", "model.weights.h5"), "r")
model_61 = StackLSTM_61()

layers = [
    (model_61.lstm1, r"_layer_checkpoint_dependencies\bidirectional"),
    (model_61.lstm2, r"_layer_checkpoint_dependencies\bidirectional_1"),
    (model_61.lstm3, r"_layer_checkpoint_dependencies\bidirectional_2"),
    (model_61.lstm4, r"_layer_checkpoint_dependencies\bidirectional_3"),
    (model_61.lstm5, r"_layer_checkpoint_dependencies\bidirectional_4"),
]
for mod, grp_name in layers:
    load_bilstm(
        mod,
        f61[grp_name + r"\forward_layer\cell"],
        f61[grp_name + r"\backward_layer\cell"]
    )

fc_w_61 = f61[r"_layer_checkpoint_dependencies\dense"]["vars"]["0"][:] # (128, 61)
fc_b_61 = f61[r"_layer_checkpoint_dependencies\dense"]["vars"]["1"][:]
model_61.fc.weight.data = torch.tensor(fc_w_61, dtype=torch.float32).t()
model_61.fc.bias.data = torch.tensor(fc_b_61, dtype=torch.float32)

f61.close()

pt_61_path = os.path.join(BASE_DIR, "stack_lstm_61.pt")
torch.save(model_61.state_dict(), pt_61_path)
print(f"[+] StackLSTM successfully saved to {pt_61_path}!")

model_61.eval()
with torch.no_grad():
    dummy = torch.randn(1, 50, 225)
    out61 = model_61(dummy)
    print(f"[+] StackLSTM test inference output: shape={out61.shape}, sum={out61.sum().item():.4f}")

print("\n" + "=" * 60)
print("[SUCCESS] Both models converted to PyTorch flawlessly!")
print("=" * 60)
