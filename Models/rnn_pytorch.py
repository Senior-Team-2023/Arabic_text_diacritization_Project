import torch
import torch.nn as nn


# Define the RNN model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden_size, 64),nn.Tanh(),nn.Linear(64, output_size))

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        # reshape the output to be able to pass it to the linear layer
        # output = output.contiguous().view(-1, self.hidden_size)
        output = self.out(output)
        return output

    def init_hidden(self, batch_size):
        return torch.zeros(1,batch_size, self.hidden_size)


# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # Use LSTM instead of RNN
        self.out = nn.Sequential(nn.Linear(hidden_size, 64),nn.Tanh(),nn.Linear(64, output_size))

    def forward(self, input, hidden):
        output, (hidden, cell) = self.lstm(input, hidden)  # LSTM returns output, (hidden state, cell state)
        output = self.out(output)
        return output

    def init_hidden(self, batch_size):
        # LSTM has two hidden states, so we initialize two hidden states
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))