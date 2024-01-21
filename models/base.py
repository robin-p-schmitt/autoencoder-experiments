from torch import nn

from abc import ABC, abstractmethod


class Model(nn.Module, ABC):
  def __init__(self):
    super(Model, self).__init__()

  @abstractmethod
  def alias(self):
    pass
