import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, output_size, num_layers):
        super().__init__()
        self.backbone = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # utilize the GRU model in torch.nn
        self.fc = nn.Linear(seq_len, output_size)

    def forward(self, x):
        x, _ = self.backbone(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        
        return x
    
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, output_size, num_layers):
        super().__init__()
        self.backbone = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(seq_len, output_size)

    def forward(self, x):
        x, _ = self.backbone(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        
        return x

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, output_size, num_layers):
        super().__init__()
        self.backbone = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) 
        self.fc = nn.Linear(seq_len, output_size)

    def forward(self, x):
        x, _ = self.backbone(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x
