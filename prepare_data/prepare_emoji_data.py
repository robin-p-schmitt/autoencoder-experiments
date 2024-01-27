import numpy as np

from prepare_data.base import DataPreparer


def main():
  """
  Filter data and copy to data/emojis/image/cleaned.
  :return:
  """

  # image_names = [f"{i}.png" for i in range(1, 1328)]  # all emojis without symbols and flags
  image_names = [
    f"{i}.png" for i in range(1, 98) if i not in (53, 56, 57, 93, 95, 96, 97)]  # all mostly yellow emoji faces
  num_images = len(image_names)
  train_size = int(0.9 * num_images)
  dev_size = num_images - train_size

  target_classes = ("Apple", "Google", "Facebook", "Samsung", "Twitter", "Windows")
  train_image_names = {}
  dev_image_names = {}
  devtrain_image_names = {}
  for class_ in target_classes:
    np.random.shuffle(image_names)
    train_image_names[class_] = image_names[:train_size]
    dev_image_names[class_] = image_names[train_size:]
    devtrain_image_names[class_] = train_image_names[class_][:dev_size]

  data_preparer = DataPreparer(
    source_data_path="data/emojis/image",
    target_data_path="data/emojis/image/cleaned",
    dev_image_names=dev_image_names,
    devtrain_image_names=devtrain_image_names,
    target_classes=target_classes,
    filter_img_size=(72, 72),
    filter_names=image_names
  )
  data_preparer.prepare()


if __name__ == '__main__':
  main()
