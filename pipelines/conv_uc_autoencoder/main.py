import torch

from models.conv_uc_autoencoder import ConvUcAE
from data_loader import ImageData

from pipelines.train import Trainer


def run_conv_uc_ae():
  train_data = ImageData(
    image_folder="data/emojis/image/cleaned/train",
    batch_size=64,
    img_size=(72, 72),
    resize_factor=2,
    augment_preset="v2",
  )
  dev_data = ImageData(
    image_folder="data/emojis/image/cleaned/dev",
    batch_size=32,
    img_size=(72, 72),
    resize_factor=2,
    augment_preset="v1",
    shuffle=False,
  )
  model = ConvUcAE(input_size=train_data.img_size, num_channels=train_data.num_channels, hidden_size=512).cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  num_epochs = 10

  trainer = Trainer(train_data, dev_data, model, optimizer, num_epochs)
  trainer.train()
