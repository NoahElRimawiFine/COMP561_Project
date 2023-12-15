import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, num_kernels, kernel_width, lstm_layers, seq_len, dropout=0.5):
        super(CNN_LSTM, self).__init__()
        self.num_kernels = num_kernels
        self.kernel_width = kernel_width
        self.lstm_layers = lstm_layers
        self.seq_len = seq_len

        # Convolutional Layer
        self.conv = nn.Conv1d(in_channels=1, out_channels=num_kernels, kernel_size=kernel_width)
        self.relu = nn.ReLU()

        # LSTM Layers
        lstm_input_sizes = [num_kernels] + lstm_layers[:-1]
        lstm_output_sizes = lstm_layers
        self.lstm_modules = nn.ModuleList([
            nn.LSTM(input_size=in_size, hidden_size=out_size, batch_first=True)
            for in_size, out_size in zip(lstm_input_sizes, lstm_output_sizes)
        ])
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(lstm_layers[-1], 2)

    def forward(self, x):
        # Apply Convolutional Layer
        x = self.conv(x)
        x = self.relu(x)

        # Adjust the output for LSTM
        x = x.transpose(1, 2)

        # Apply LSTM layers
        for lstm in self.lstm_modules:
            x, _ = lstm(x)
            x = self.dropout(x)

        # Select the last time step's output
        x = x[:, -1, :]

        # Fully connected layer
        x = self.fc(x)
        return x