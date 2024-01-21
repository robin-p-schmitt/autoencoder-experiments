import os

import torch
from torch import nn


def train(data, model, optimizer, num_epochs):
  saved_model_path = f'checkpoints/{model.alias()}_epoch_{num_epochs}.pth'
  if os.path.exists(saved_model_path):
    print("Model already trained.")
    return saved_model_path

  print("Start training...")
  for epoch in range(num_epochs):
    for batch in data.data_loader:
      # flatten images and move to device
      images = batch[0].cuda()

      # forward pass and loss
      outputs = model(images)
      loss = nn.MSELoss()(outputs, images.view(images.size(0), -1))

      # backward pass and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # print loss
      print(f'epoch [{epoch + 1}/{num_epochs}], batch loss:{loss.item():.4f}')

  # save model
  torch.save(model.state_dict(), saved_model_path)
  return saved_model_path