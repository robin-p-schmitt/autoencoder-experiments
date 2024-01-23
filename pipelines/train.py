import os

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

from data_loader import ImageData


class Trainer:
  def __init__(
          self,
          train_data: ImageData,
          dev_data: ImageData,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          num_epochs: int
  ):
    self.train_data = train_data
    self.dev_data = dev_data
    self.model = model
    self.optimizer = optimizer
    self.num_epochs = num_epochs

    self.num_train_batches = len(self.train_data.data_loader)
    self.num_dev_batches = len(self.dev_data.data_loader)

    self.saved_model_path = self._get_saved_model_path()

    # create folder for example reconstructions
    saved_model_folder = os.path.dirname(self.saved_model_path)
    self.visualizations_folder = os.path.join(saved_model_folder, "visualizations")
    if not os.path.exists(self.visualizations_folder):
      os.makedirs(self.visualizations_folder)

    # create tensorboard writer
    self.writer = SummaryWriter(log_dir=os.path.join(saved_model_folder, "tensorboard"))

    # create list of sample inputs and dict of sample outputs
    self.num_samples = 10
    self.sample_inputs = []
    self.sample_outputs = {i: [] for i in range(self.num_epochs)}

  def train(self):
    if os.path.exists(self.saved_model_path):
      print("Model already trained.")
      return

    print("Start training...")
    for epoch in range(self.num_epochs):
      self._train_loop(epoch)
      self._dev_loop(epoch)

    torch.save(self.model.state_dict(), self.saved_model_path)
    self.writer.close()
    self._save_visualizations()

  def _get_saved_model_path(self):
    # create folder for data alias
    data_folder = os.path.join("checkpoints", self.train_data.alias())
    if not os.path.exists(data_folder):
      os.makedirs(data_folder)

    # create folder for model alias
    saved_model_folder = os.path.join(data_folder, self.model.alias())
    if not os.path.exists(saved_model_folder):
      os.makedirs(saved_model_folder)

    # save model in the data alias folder
    saved_model_path = os.path.join(saved_model_folder, f'model_epoch_{self.num_epochs}.pth')

    return saved_model_path

  def _train_loop(self, epoch: int):
    accum_loss = 0.0
    for i, batch in enumerate(self.train_data.data_loader):
      # flatten images and move to device
      images = batch[0].cuda()

      # forward pass and loss
      outputs = self.model(images)
      loss = nn.MSELoss()(outputs.view(outputs.size(0), -1), images.view(images.size(0), -1))
      accum_loss += loss.item()

      # backward pass and optimize
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # print and log loss
      self.writer.add_scalar('Loss/train', loss.item(), epoch * self.num_train_batches + i)

    avg_loss = accum_loss / self.num_train_batches
    print(f'epoch [{epoch + 1}/{self.num_epochs}], loss:{avg_loss:.4f}')

  def _dev_loop(self, epoch: int):
    # evaluate on dev set
    with torch.no_grad():
      for i, batch in enumerate(self.dev_data.data_loader):
        # flatten images and move to device
        images = batch[0].cuda()

        # forward pass and loss
        outputs = self.model(images)
        loss = nn.MSELoss()(outputs.view(outputs.size(0), -1), images.view(images.size(0), -1))

        # print and log loss
        self.writer.add_scalar('Loss/dev', loss.item(), epoch * self.num_dev_batches + i)

        # save example reconstructions
        if i == 0:
          for j in range(self.num_samples):
            sample_input = batch[0][j].permute(1, 2, 0).cpu()
            sample_output = outputs[j].permute(1, 2, 0).cpu()
            if epoch == 0:
              self.sample_inputs.append(sample_input)
            self.sample_outputs[epoch].append(sample_output)

  def _save_visualizations(self):
    # save example reconstructions
    fig, axarray = plt.subplots(self.num_samples, self.num_epochs + 1)
    for i in range(self.num_samples):
      axarray[i, 0].imshow(self.sample_inputs[i])
      for epoch in range(self.num_epochs):
        axarray[i, epoch + 1].imshow(self.sample_outputs[epoch][i])
    figure_path = os.path.join(self.visualizations_folder, f"samples.png")
    plt.savefig(figure_path)
    plt.close(fig)
