import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple
import numpy as np

from models.base import Model


class ConvUcAE(Model):
  def __init__(self, input_size: Tuple[int, int, int], num_channels: int, hidden_size: int):
    super(ConvUcAE, self).__init__()
    assert input_size[0] == num_channels, "expected data layout: (num_channels, height, width)"

    self.hidden_size = hidden_size

    # Encoder
    self.conv1 = nn.Conv2d(num_channels, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
    self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
    self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)

    # fully-connected
    self.processed_size = (64, input_size[1] // 4, input_size[2] // 4)
    flattened_len = np.prod(self.processed_size)
    self.fc1 = nn.Linear(flattened_len, hidden_size)
    self.fc2 = nn.Linear(hidden_size, flattened_len)

    # Decoder
    self.t_conv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
    self.t_conv2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
    self.t_conv3 = nn.ConvTranspose2d(16, 16, 3, padding=1)
    self.t_conv4 = nn.ConvTranspose2d(16, num_channels, 3, padding=1)

  def forward(self, x):
    # convolutions
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))

    x = self.pool(x)
    x = F.relu(self.conv3(x))
    x = self.pool(x)
    x = F.relu(self.conv4(x))

    # fully-connected
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = x.view(x.size(0), *self.processed_size)

    # transposed convolutions
    x = F.relu(self.t_conv1(x))
    x = F.relu(self.t_conv2(x))
    x = F.relu(self.t_conv3(x))
    x = F.sigmoid(self.t_conv4(x))
    return x

  def alias(self):
    return f"conv_uc_ae_{self.hidden_size}"


class ConvUcAE2(Model):
  def __init__(self, input_size: Tuple[int, int, int], num_channels: int, hidden_size: int):
    super(ConvUcAE2, self).__init__()
    assert input_size[0] == num_channels, "expected data layout: (num_channels, height, width)"

    self.hidden_size = hidden_size

    # Encoder
    self.conv1 = nn.Conv2d(num_channels, 64, 3, padding=1, stride=2)
    self.conv2 = nn.Conv2d(64, 32, 3, padding=1, stride=2)
    self.conv3 = nn.Conv2d(32, 16, 3, padding=1, stride=2)
    self.conv4 = nn.Conv2d(16, 8, 3, padding=1, stride=1)

    # fully-connected
    self.processed_size = (8, input_size[1] // 8, input_size[2] // 8)
    flattened_len = np.prod(self.processed_size)
    self.fc1 = nn.Linear(flattened_len, hidden_size)
    self.fc2 = nn.Linear(hidden_size, flattened_len)

    # Decoder
    self.t_conv1 = nn.ConvTranspose2d(8, 16, 3, padding=1, stride=1)
    self.t_conv2 = nn.ConvTranspose2d(16, 32, 3, padding=1, stride=2, output_padding=1)
    self.t_conv3 = nn.ConvTranspose2d(32, 64, 3, padding=1, stride=2, output_padding=1)
    self.t_conv4 = nn.ConvTranspose2d(64, num_channels, 3, padding=1, stride=2, output_padding=1)

  def forward(self, x):
    # convolutions
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))

    # fully-connected
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = x.view(x.size(0), *self.processed_size)

    # transposed convolutions
    x = F.relu(self.t_conv1(x))
    x = F.relu(self.t_conv2(x))
    x = F.relu(self.t_conv3(x))
    x = F.sigmoid(self.t_conv4(x))
    return x

  def alias(self):
    return f"conv_uc_ae2_{self.hidden_size}"
