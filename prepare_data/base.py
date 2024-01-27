import os
from PIL import Image
from typing import Optional, Dict, List, Tuple


class DataPreparer:
  def __init__(
          self,
          source_data_path: str,
          target_data_path: str,
          dev_image_names: Dict[str, List[str]],
          devtrain_image_names: Dict[str, List[str]],
          target_classes: Tuple[str, ...],
          filter_img_size: Optional[Tuple[int, int]] = None,
          filter_names: Optional[List[str]] = None,
  ):
    self.filter_img_size = filter_img_size
    self.target_classes = target_classes
    self.filter_names = filter_names
    self.source_data_path = source_data_path
    self.target_data_path = target_data_path
    self.dev_image_names = dev_image_names
    self.devtrain_image_names = devtrain_image_names

  def prepare(self):
    DataPreparer.create_path(self.target_data_path)

    train_path = DataPreparer.create_path(os.path.join(self.target_data_path, "train"))
    dev_path = DataPreparer.create_path(os.path.join(self.target_data_path, "dev"))
    devtrain_path = DataPreparer.create_path(os.path.join(self.target_data_path, "devtrain"))

    for target_class in self.target_classes:
      source_class_path = os.path.join(self.source_data_path, target_class)

      train_class_path = DataPreparer.create_path(os.path.join(train_path, target_class))
      dev_class_path = DataPreparer.create_path(os.path.join(dev_path, target_class))
      devtrain_class_path = DataPreparer.create_path(os.path.join(devtrain_path, target_class))

      for file_name in os.listdir(source_class_path):
        if not self.filter_by_name(file_name):
          continue

        img = Image.open(os.path.join(source_class_path, file_name))
        if not self.filter_by_image_size(img):
          continue

        if file_name in self.dev_image_names[target_class]:
          img.save(os.path.join(dev_class_path, file_name))
        else:
          img.save(os.path.join(train_class_path, file_name))
          if file_name in self.devtrain_image_names[target_class]:
            img.save(os.path.join(devtrain_class_path, file_name))

  @staticmethod
  def create_path(path: str):
    if not os.path.exists(path):
      os.makedirs(path)
    return path

  def filter_by_image_size(self, img: Image) -> bool:
    if self.filter_img_size is None:
      return True

    return img.size == self.filter_img_size

  def filter_by_name(self, path: str) -> bool:
    if self.filter_names is None:
      return True
    return os.path.basename(path) in self.filter_names
