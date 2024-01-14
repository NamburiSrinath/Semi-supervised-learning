import torch
import torch.nn as nn

class SmallNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SmallNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x