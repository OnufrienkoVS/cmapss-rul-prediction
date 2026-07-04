import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    """LSTM модель для прогнозирования RUL."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_size* 2, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, window_size, input_size)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNN1D(nn.Module):
    def __init__(self, input_size, window_size, num_filters=64, kernel_size=3, dropout=0.3):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size + 4, padding='same')
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size + 2, padding='same')
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size, padding='same')
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        self.pool3 = nn.MaxPool1d(2)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters*4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, window_size, input_size), нужно поменять местами каналы (input_size) и последовательности (window_size)
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def get_model_class(model_type: str):
    """Возвращает класс модели по типу."""
    if model_type == "lstm":
        return LSTMModel
    elif model_type == "cnn":
        return CNN1D
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")