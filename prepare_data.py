import os
from PIL import Image


def main():
  """
  Currently, we just use the Apple emojis and copy images 0-95 (all face emojis)
  with size 72x72 into data/emojis/image/cleaned.
  :return:
  """
  data_path = "data/emojis/image"
  target_classes = ("Apple",)
  target_path = os.path.join(data_path, "cleaned")
  if not os.path.exists(target_path):
    os.makedirs(target_path)

  for target_class in target_classes:
    source_class_path = os.path.join(data_path, target_class)
    target_class_path = os.path.join(target_path, target_class)
    if not os.path.exists(target_class_path):
      os.makedirs(target_class_path)

    for path in os.listdir(source_class_path):
      img = Image.open(os.path.join(source_class_path, path))
      if img.size == (72, 72) and (int(os.path.splitext(path)[0]) <= 95):
        img.save(os.path.join(target_class_path, path))


if __name__ == '__main__':
  main()
