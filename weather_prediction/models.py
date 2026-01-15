import torch
from torch.nn import nn

class Weather3DCNN(nn.Module):
    def __init__(self, input_channels = 5, input_frames = 20):
        super().__init__()
        