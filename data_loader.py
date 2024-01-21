from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from typing import Tuple

from PIL import Image


def custom_loader(path):
  img = Image.open(path)
  return img.convert('RGBA')


class ImageData:
  def __init__(self, image_folder: str, batch_size: int, img_size: Tuple[int, int], alpha: bool = True):
    self.image_folder = image_folder
    self.img_size = (4,) + img_size if alpha else (3,) + img_size

    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = datasets.ImageFolder(root=self.image_folder, transform=transform, loader=custom_loader)
    self.data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
