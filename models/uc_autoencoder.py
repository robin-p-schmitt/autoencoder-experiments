import torch
from torch import nn
from typing import Tuple
import numpy as np

from models.base import Model


class ShallowFFUcAE(Model):
  def __init__(self, input_size: Tuple[int, int, int], hidden_size: int):
    super(ShallowFFUcAE, self).__init__()
    self.input_size = np.prod(input_size)
    self.hidden_size = hidden_size

    self.encoder = nn.Sequential(
      nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
      nn.ReLU()
    )
    self.decoder = nn.Sequential(
      nn.Linear(in_features=self.hidden_size, out_features=self.input_size),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def alias(self):
    return f"shallow_ff_uc_ae_{self.input_size}_{self.hidden_size}"


class DeepFFUcAE(ShallowFFUcAE):
  def __init__(self, input_size: Tuple[int, int, int], layer_sizes: Tuple[int, ...]):
    super(DeepFFUcAE, self).__init__(input_size, layer_sizes[-1])
    self.layer_sizes = layer_sizes

    # encoder
    encoder_layers = [nn.Linear(in_features=self.input_size, out_features=layer_sizes[0]), nn.ReLU()]
    for i in range(0, len(layer_sizes) - 1):
      encoder_layers.append(nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i + 1]))
      encoder_layers.append(nn.ReLU())
    self.encoder = nn.Sequential(*encoder_layers)

    # decoder
    decoder_layers = []
    for i in range(len(layer_sizes) - 1, 0, -1):
      decoder_layers.append(nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i - 1]))
      decoder_layers.append(nn.ReLU())
    decoder_layers += [nn.Linear(in_features=layer_sizes[0], out_features=self.input_size), nn.Sigmoid()]
    self.decoder = nn.Sequential(*decoder_layers)

  def alias(self):
    return f"deep_ff_uc_ae_{self.input_size}_{'_'.join(map(str, self.layer_sizes))}"
