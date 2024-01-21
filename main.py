import torch
from torch import nn
from matplotlib import pyplot as plt
import math
import os

from models.uc_autoencoder import UndercompleteAutoencoder
from data_loader import ImageData


def train(data, model, optimizer, num_epochs):
  saved_model_path = f'checkpoints/{model.alias()}_epoch_{num_epochs}.pth'
  if os.path.exists(saved_model_path):
    print("Model already trained.")
    return saved_model_path

  print("Start training...")
  for epoch in range(num_epochs):
    for batch in data.data_loader:
      # flatten images and move to device
      images = batch[0].view(batch[0].size(0), -1).cuda()

      # forward pass and loss
      outputs = model(images)
      loss = nn.MSELoss()(outputs, images)

      # backward pass and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # print loss
      print(f'epoch [{epoch + 1}/{num_epochs}], batch loss:{loss.item():.4f}')

  # save model
  torch.save(model.state_dict(), saved_model_path)
  return saved_model_path


def inference(data, model, saved_model_path):
  # load model from checkpoint
  if not os.path.exists(saved_model_path):
    print("Model not found, aborting inference.")
    return
  model.load_state_dict(torch.load(saved_model_path))

  # visualize some images
  with torch.no_grad():
    for batch in data.data_loader:
      images = batch[0].view(batch[0].size(0), -1).cuda()
      outputs = model(images)
      outputs = outputs.view(outputs.size(0), *data.img_size)
      for i in range(5):
        # show original image and reconstructed image next to each other
        fig, axarray = plt.subplots(1, 2)
        axarray[0].imshow(batch[0][i].permute(1, 2, 0).cpu())
        axarray[1].imshow(outputs[i].cpu().permute(1, 2, 0).cpu())
        plt.show()
      break


def main():
  image_data = ImageData(image_folder="data/emojis/image/cleaned", batch_size=32, img_size=(72, 72))
  model = UndercompleteAutoencoder(input_size=math.prod(image_data.img_size), hidden_size=100).cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  num_epochs = 20

  saved_model_path = train(image_data, model, optimizer, num_epochs)
  inference(image_data, model, saved_model_path)


if __name__ == '__main__':
  main()
