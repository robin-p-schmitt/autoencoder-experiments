import torch

from models.ff_uc_autoencoder import ShallowFFUcAE, DeepFFUcAE
from data_loader import ImageData

from pipelines.train import Trainer


def run_shallow_ff_uc_ae():
  train_data = ImageData(image_folder="data/emojis/image/cleaned/train", batch_size=32, img_size=(72, 72))
  dev_data = ImageData(image_folder="data/emojis/image/cleaned/dev", batch_size=32, img_size=(72, 72))
  model = ShallowFFUcAE(input_size=train_data.img_size, hidden_size=256).cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  num_epochs = 30

  trainer = Trainer(train_data, dev_data, model, optimizer, num_epochs)
  trainer.train()


def run_deep_ff_uc_ae():
  train_data = ImageData(image_folder="data/emojis/image/cleaned", batch_size=32, img_size=(72, 72))
  dev_data = ImageData(image_folder="data/emojis/image/cleaned/dev", batch_size=32, img_size=(72, 72))
  model = DeepFFUcAE(input_size=train_data.img_size, layer_sizes=(1024, 512, 256)).cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  num_epochs = 30

  trainer = Trainer(train_data, dev_data, model, optimizer, num_epochs)
  trainer.train()
