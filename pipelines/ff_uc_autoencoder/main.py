import torch

from models.ff_uc_autoencoder import ShallowFFUcAE, DeepFFUcAE
from data_loader import ImageData

from pipelines.train import train
from pipelines.inference import inference


def run_shallow_ff_uc_ae():
  train_data = ImageData(image_folder="data/emojis/image/cleaned/train", batch_size=32, img_size=(72, 72))
  dev_data = ImageData(image_folder="data/emojis/image/cleaned/dev", batch_size=32, img_size=(72, 72))
  model = ShallowFFUcAE(input_size=train_data.img_size, hidden_size=256).cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  num_epochs = 30

  saved_model_path = train(train_data, model, optimizer, num_epochs)
  inference(dev_data, model, saved_model_path)


def run_deep_ff_uc_ae():
  train_data = ImageData(image_folder="data/emojis/image/cleaned", batch_size=32, img_size=(72, 72))
  dev_data = ImageData(image_folder="data/emojis/image/cleaned/dev", batch_size=32, img_size=(72, 72))
  model = DeepFFUcAE(input_size=train_data.img_size, layer_sizes=(1024, 512, 256)).cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  num_epochs = 30

  saved_model_path = train(train_data, model, optimizer, num_epochs)
  inference(dev_data, model, saved_model_path)
