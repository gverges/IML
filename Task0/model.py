import torch
import torch.nn as nn
import torch.optim as optim

class CNNRegressionModel(nn.Module):
    def __init__(self, input_shape):
        super(CNNRegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * input_shape[1] * input_shape[2], 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_cnn_regression_model(input_shape):
    model = CNNRegressionModel(input_shape)
    criterion = lambda output, target: torch.sqrt(nn.MSELoss()(output, target))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer