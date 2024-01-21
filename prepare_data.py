import os
from PIL import Image
from typing import Tuple, Optional


class DataPreparer:
  def __init__(
          self,
          source_data_path: str,
          target_data_path: str,
          dev_image_names: Tuple[str, ...],
          target_classes: Tuple[str, ...],
          filter_img_size: Optional[Tuple[int, int]] = None,
          filter_names: Optional[Tuple[str, ...]] = None,
  ):
    self.filter_img_size = filter_img_size
    self.target_classes = target_classes
    self.filter_names = filter_names
    self.source_data_path = source_data_path
    self.target_data_path = target_data_path
    self.dev_image_names = dev_image_names

  def prepare(self):
    if not os.path.exists(self.target_data_path):
      os.makedirs(self.target_data_path)

    train_path = os.path.join(self.target_data_path, "train")
    if not os.path.exists(train_path):
      os.makedirs(train_path)

    dev_path = os.path.join(self.target_data_path, "dev")
    if not os.path.exists(dev_path):
      os.makedirs(dev_path)

    for target_class in self.target_classes:
      source_class_path = os.path.join(self.source_data_path, target_class)
      train_class_path = os.path.join(train_path, target_class)
      if not os.path.exists(train_class_path):
        os.makedirs(train_class_path)
      dev_class_path = os.path.join(dev_path, target_class)
      if not os.path.exists(dev_class_path):
        os.makedirs(dev_class_path)

      for path in os.listdir(source_class_path):
        if not self.filter_by_name(path):
          continue

        img = Image.open(os.path.join(source_class_path, path))
        if not self.filter_by_image_size(img):
          continue

        if path in self.dev_image_names:
          img.save(os.path.join(dev_class_path, path))
        else:
          img.save(os.path.join(train_class_path, path))

  def filter_by_image_size(self, img: Image) -> bool:
    if self.filter_img_size is None:
      return True

    return img.size == self.filter_img_size

  def filter_by_name(self, path: str) -> bool:
    if self.filter_names is None:
      return True
    return os.path.basename(path) in self.filter_names


def main():
  """
  Filter data and copy to data/emojis/image/cleaned.
  :return:
  """

  outlier_names = [38, 53, 56, 57, 61, 79, 80, 84, 93, 95]  # outlier faces, e.g. not yellow
  filter_names = [i for i in range(96) if i not in outlier_names]  # non-outlier face emojis

  data_preparer = DataPreparer(
    source_data_path="data/emojis/image",
    target_data_path="data/emojis/image/cleaned",
    dev_image_names=tuple([f"{i}.png" for i in [3, 20, 29, 36]]),
    target_classes=("Apple",),
    filter_img_size=(72, 72),
    filter_names=tuple([f"{i}.png" for i in filter_names]),
  )
  data_preparer.prepare()


if __name__ == '__main__':
  main()
