import torch

from models.conv_uc_autoencoder import ConvUcAE, ConvUcAE2
from data_loader import ImageData

from pipelines.train import Trainer


def run_conv_uc_ae():
  train_data = ImageData(
    image_folder="data/emojis/image/cleaned/train",
    batch_size=64,
    img_size=(72, 72),
    resize_factor=1,
    augment_preset="v4",
  )
  dev_data = ImageData(
    image_folder="data/emojis/image/cleaned/dev",
    batch_size=32,
    img_size=(72, 72),
    resize_factor=1,
    augment_preset="v1",
    shuffle=False,
  )
  devtrain_data = ImageData(
    image_folder="data/emojis/image/cleaned/devtrain",
    batch_size=32,
    img_size=(72, 72),
    resize_factor=1,
    augment_preset="v1",
    shuffle=False,
  )
  model = ConvUcAE2(input_size=train_data.img_size, num_channels=train_data.num_channels, hidden_size=128).cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  num_epochs = 200

  num_epoch_samples = 10
  epoch_sample_interval = num_epochs // num_epoch_samples
  sample_epochs = [i * epoch_sample_interval for i in range(num_epoch_samples)]

  trainer = Trainer(
    train_data,
    dev_data,
    devtrain_data,
    model,
    optimizer,
    num_epochs,
    sample_epochs,
    False
  )
  trainer.train()
