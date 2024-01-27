from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt

from data_loader import ImageData


class TensorTransform(ABC):
    """Base class for all tensor transformations."""

    @abstractmethod
    def __call__(self, tensor):
        raise NotImplementedError


class DoGFilter(TensorTransform):
    """Difference of Gaussian filter."""

    def __init__(self, kernel_size: int, sigma1: float, sigma2: float, cuda: bool = True):
        self.kernel_size = kernel_size
        self._dog_filter = DoGFilter.dog_filter(kernel_size, sigma1, sigma2)
        if cuda:
            self._dog_filter = self._dog_filter.cuda()

    @staticmethod
    def gaussian_filter_1d(kernel_size, sigma):
        """Create a 1D Gaussian kernel."""
        kernel = torch.arange(kernel_size).float() - kernel_size // 2
        kernel = torch.exp(-kernel ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    @staticmethod
    def gaussian_filter_2d(kernel_size, sigma):
        """Create a 2D Gaussian kernel."""
        kernel_1d = DoGFilter.gaussian_filter_1d(kernel_size, sigma)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d

    @staticmethod
    def dog_filter(kernel_size, sigma1, sigma2):
        """Create a 2D Difference of Gaussian kernel."""
        kernel1 = DoGFilter.gaussian_filter_2d(kernel_size, sigma1)
        kernel2 = DoGFilter.gaussian_filter_2d(kernel_size, sigma2)
        kernel = kernel1 - kernel2
        return kernel.expand(1, 4, kernel_size, kernel_size)

    def __call__(self, tensor):
        """Apply DoG filter to tensor."""
        return F.conv2d(tensor, self._dog_filter, padding=self.kernel_size // 2)


# plot example visualizations when run as a script
if __name__ == "__main__":
  dev_data = ImageData(
    image_folder="data/emojis/image/cleaned/dev",
    batch_size=32,
    img_size=(72, 72),
    resize_factor=1,
    augment_preset="v1",
    shuffle=False,
  )
  dog_filters = [
    DoGFilter(kernel_size=3, sigma1=1.0, sigma2=2.0, cuda=False),
    DoGFilter(kernel_size=5, sigma1=1.0, sigma2=2.0, cuda=False),
    DoGFilter(kernel_size=11, sigma1=2.0, sigma2=3.0, cuda=False),
  ]

  dog_batches = []
  for batch in dev_data.data_loader:
    for dog_filter in dog_filters:
      dog_batches.append(dog_filter(batch[0]))

    for i in range(2):
      fig, ax = plt.subplots(1, len(dog_filters) + 1)
      ax[0].imshow(batch[0][i].permute(1, 2, 0))
      for j, dog_batch in enumerate(dog_batches):
        ax[j + 1].imshow(dog_batch[i][0].detach().numpy())
      plt.show()
      plt.close(fig)
    break
