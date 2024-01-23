from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from typing import Tuple, Optional, Dict

from PIL import Image


augment_opts_presets = {
  "v1": None,
  "v2": {
    "rot": 30,
    "vert_flip": 0.5,
    "crop": {"scale": (0.3, 1.0), "ratio": (1.0, 1.0)},
  },
  "v3": {
    "rot": 30,
    "color": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.3},
    "vert_flip": 0.5,
    "crop": {"scale": (0.3, 1.0), "ratio": (1.0, 1.0)},
  },
  "v4": {
    "rot": 30,
    "color": {"saturation": 0.3, "hue": 0.3},
    "vert_flip": 0.5,
    "crop": {"scale": (0.3, 1.0), "ratio": (1.0, 1.0)},
  }
}


def custom_loader(path):
  img = Image.open(path)
  return img.convert('RGBA')


class ImageData:
  def __init__(
          self,
          image_folder: str,
          batch_size: int,
          img_size: Tuple[int, int],
          alpha: bool = True,
          resize_factor: Optional[int] = None,
          augment_preset: str = "v1",
          shuffle: bool = True,
  ):
    self.image_folder = image_folder
    self.num_channels = 4 if alpha else 3
    self.img_size = (self.num_channels,) + img_size
    self.resize_factor = resize_factor
    self.batch_size = batch_size
    if resize_factor is not None:
      self.img_size = (self.img_size[0], self.img_size[1] // resize_factor, self.img_size[2] // resize_factor)

    # DATA TRANSFORMS
    # first resize
    transform = [transforms.Resize(self.img_size[1:])]

    # then augment
    self.augment_preset = augment_preset
    augment_opts = augment_opts_presets[augment_preset]
    if augment_opts is not None:
      if "rot" in augment_opts:
        transform.append(transforms.RandomRotation(augment_opts["rot"]))
      if "color" in augment_opts:
        transform.append(transforms.ColorJitter(**augment_opts["color"]))
      if "vert_flip" in augment_opts:
        transform.append(transforms.RandomVerticalFlip(augment_opts["vert_flip"]))
      if "gauss" in augment_opts:
        transform.append(transforms.GaussianBlur(**augment_opts["gauss"]))
      if "crop" in augment_opts:
        transform.append(transforms.RandomResizedCrop(self.img_size[1:], **augment_opts["crop"]))
    # then convert to tensor and normalize
    transform += [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    transform = transforms.Compose(transform)
    dataset = datasets.ImageFolder(root=self.image_folder, transform=transform, loader=custom_loader)

    self.data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

  def alias(self):
    alias = ""
    alias += f"batch-size-{self.batch_size}_"
    if self.resize_factor is not None:
      alias += f"resize_{self.resize_factor}_"
    alias += f"augment-{self.augment_preset}"

    return alias
