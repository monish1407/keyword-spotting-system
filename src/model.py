import torch
import torch.nn as nn


class KeywordLSTM(nn.Module):
    """
    LSTM model for keyword classification.
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(KeywordLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out