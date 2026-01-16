import torch
import torch.nn as nn
import numpy as np

class Weather3DCNN(nn.Module):
    def __init__(self, input_channels = 5, input_frames = 20):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1, stride = (1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1, stride = (2, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1, stride = (2, 2, 2))
        self.bn3 = nn.BatchNorm3d(128)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.pool(x)
        x = self.fc(x)
        return x
    
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
        
if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = Weather3DCNN(input_channels=5, input_frames=20).to(device)