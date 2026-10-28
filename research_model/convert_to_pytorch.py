import os
import h5py
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = r"d:\Final Project\research_model"

print("=" * 65)
print("Re-converting Keras H5 Weights with Exact Gate Alignment & BiLSTM Fix")
print("=" * 65)

def convert_lstm_weights(k_kernel, k_recurrent, k_bias):
    # Keras default order is [i, f, c, o]
    # PyTorch default order is [i, f, g, o] (where g is c, the candidate cell)
    # Both are identical! We directly transpose without swapping gates.
    w_ih = torch.tensor(k_kernel.T, dtype=torch.float32)
    w_hh = torch.tensor(k_recurrent.T, dtype=torch.float32)
    b_ih = torch.tensor(k_bias, dtype=torch.float32)
    b_hh = torch.zeros_like(b_ih)
    return w_ih, w_hh, b_ih, b_hh

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
        # Keras merge_mode='concat' on return_sequences=False:
        # Forward final output is at t=-1, backward final output is at t=0!
        fwd_last = out[:, -1, :64]
        bwd_last = out[:, 0, 64:]
        last_step = torch.cat([fwd_last, bwd_last], dim=-1)
        logits = self.fc(last_step)
        return self.softmax(logits)

# -------------------------------------------------------------
# 2. StackLSTM (61 Classes, 98.5% Accuracy)
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
        fwd_last = out[:, -1, :64]
        bwd_last = out[:, 0, 64:]
        last_step = torch.cat([fwd_last, bwd_last], dim=-1)
        logits = self.fc(last_step)
        return self.softmax(logits)

# Convert 17 Class Model
print("[1/2] Converting CNN+LSTM (17 Classes)...")
f17 = h5py.File(os.path.join(BASE_DIR, "CNN_LSTM_weights", "model.weights.h5"), "r")
model_17 = CNN_LSTM_17()

c1_w = f17[r"_layer_checkpoint_dependencies\conv1d"]["vars"]["0"][:]
c1_b = f17[r"_layer_checkpoint_dependencies\conv1d"]["vars"]["1"][:]
model_17.conv1.weight.data = torch.tensor(c1_w, dtype=torch.float32).permute(2, 1, 0)
model_17.conv1.bias.data = torch.tensor(c1_b, dtype=torch.float32)

c2_w = f17[r"_layer_checkpoint_dependencies\conv1d_1"]["vars"]["0"][:]
c2_b = f17[r"_layer_checkpoint_dependencies\conv1d_1"]["vars"]["1"][:]
model_17.conv2.weight.data = torch.tensor(c2_w, dtype=torch.float32).permute(2, 1, 0)
model_17.conv2.bias.data = torch.tensor(c2_b, dtype=torch.float32)

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

fc_w = f17[r"_layer_checkpoint_dependencies\dense"]["vars"]["0"][:]
fc_b = f17[r"_layer_checkpoint_dependencies\dense"]["vars"]["1"][:]
model_17.fc.weight.data = torch.tensor(fc_w, dtype=torch.float32).t()
model_17.fc.bias.data = torch.tensor(fc_b, dtype=torch.float32)

torch.save(model_17.state_dict(), os.path.join(BASE_DIR, "cnn_lstm_17.pt"))
print("[+] Successfully saved cnn_lstm_17.pt")

# Convert 61 Class Model
print("\n[2/2] Converting StackLSTM (61 Classes)...")
f61 = h5py.File(os.path.join(BASE_DIR, "StackLSTM_weights", "model.weights.h5"), "r")
model_61 = StackLSTM_61()

load_bilstm(
    model_61.lstm1,
    f61[r"_layer_checkpoint_dependencies\bidirectional\forward_layer\cell"],
    f61[r"_layer_checkpoint_dependencies\bidirectional\backward_layer\cell"]
)
load_bilstm(
    model_61.lstm2,
    f61[r"_layer_checkpoint_dependencies\bidirectional_1\forward_layer\cell"],
    f61[r"_layer_checkpoint_dependencies\bidirectional_1\backward_layer\cell"]
)
load_bilstm(
    model_61.lstm3,
    f61[r"_layer_checkpoint_dependencies\bidirectional_2\forward_layer\cell"],
    f61[r"_layer_checkpoint_dependencies\bidirectional_2\backward_layer\cell"]
)
load_bilstm(
    model_61.lstm4,
    f61[r"_layer_checkpoint_dependencies\bidirectional_3\forward_layer\cell"],
    f61[r"_layer_checkpoint_dependencies\bidirectional_3\backward_layer\cell"]
)
load_bilstm(
    model_61.lstm5,
    f61[r"_layer_checkpoint_dependencies\bidirectional_4\forward_layer\cell"],
    f61[r"_layer_checkpoint_dependencies\bidirectional_4\backward_layer\cell"]
)

fc_w61 = f61[r"_layer_checkpoint_dependencies\dense"]["vars"]["0"][:]
fc_b61 = f61[r"_layer_checkpoint_dependencies\dense"]["vars"]["1"][:]
model_61.fc.weight.data = torch.tensor(fc_w61, dtype=torch.float32).t()
model_61.fc.bias.data = torch.tensor(fc_b61, dtype=torch.float32)

torch.save(model_61.state_dict(), os.path.join(BASE_DIR, "stack_lstm_61.pt"))
print("[+] Successfully saved stack_lstm_61.pt")
print("\nConversion complete!")
