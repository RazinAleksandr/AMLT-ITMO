import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_d, hidden_d, layer_d, output_d):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_d
        self.layer_dim = layer_d

        # LSTM model 
        self.lstm = nn.LSTM(
            input_size=input_d, 
            hidden_size=hidden_d, 
            num_layers=layer_d, 
            batch_first=True
            )

        self.fc = nn.Linear(hidden_d, output_d)

    def forward(self, x):
        # init h0, c0
        h0 = torch.randn(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # lstm run
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # fc run
        out = self.fc(out[:, -1, :]) 
        return out