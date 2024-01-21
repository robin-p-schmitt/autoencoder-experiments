import torch
from torch import nn

from models.base import Model


class UndercompleteAutoencoder(Model):
  def __init__(self, input_size: int, hidden_size: int):
    super(UndercompleteAutoencoder, self).__init__()
    self.input_size = input_size
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
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def alias(self):
    return f"uc_autoencoder_{self.input_size}_{self.hidden_size}"
