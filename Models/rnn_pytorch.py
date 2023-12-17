import torch
import torch.nn as nn

# Define the RNN model
class RNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.rnn(x, h0)  # RNN output and last hidden state
        out = self.fc(out[:, -1, :])  # Pass the last time-step output of RNN to a Fully connected layer
        return out


