import os

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from data_loader import ImageData
from transformations import DoGFilter


class Trainer:
  def __init__(
          self,
          train_data: ImageData,
          dev_data: ImageData,
          devtrain_data: ImageData,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          num_epochs: int,
          sample_epochs: List[int],
          use_dog_aux_loss: bool = False,
  ):
    self.train_data = train_data
    self.dev_data = dev_data
    self.devtrain_data = devtrain_data
    self.model = model
    self.optimizer = optimizer
    self.num_epochs = num_epochs
    self.sample_epochs = sample_epochs
    self.use_dog_aux_loss = use_dog_aux_loss

    self.dog_filter = DoGFilter(3, 1.0, 2.0) if use_dog_aux_loss else None

    self.num_batches = {
      "train": len(self.train_data.data_loader),
      "dev": len(self.dev_data.data_loader),
      "devtrain": len(self.devtrain_data.data_loader),
    }

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
    self.sample_inputs = {"dev": [], "devtrain": []}
    self.sample_outputs = {"dev": {i: [] for i in self.sample_epochs}, "devtrain": {i: [] for i in self.sample_epochs}}

  def train(self):
    if os.path.exists(self.saved_model_path):
      print("Model already trained.")
      return

    print("Start training...")
    for epoch in range(self.num_epochs):
      train_loss = self._train_loop(epoch)
      dev_loss = self._dev_loop(epoch, self.dev_data, "dev")
      devtrain_loss = self._dev_loop(epoch, self.devtrain_data, "devtrain")

      print(
        f"epoch [{epoch + 1}/{self.num_epochs}], ",
        f"train loss:{train_loss:.4f}, ",
        f"dev loss:{dev_loss:.4f}, ",
        f"devtrain loss:{devtrain_loss:.4f}"
      )

    torch.save(self.model.state_dict(), self.saved_model_path)
    self.writer.close()
    self._save_visualizations()

  def _get_saved_model_path(self):
    # create folder for data alias
    data_folder = os.path.join("checkpoints", self.train_data.alias())
    if not os.path.exists(data_folder):
      os.makedirs(data_folder)

    # create folder for model alias
    model_alias = f"{self.model.alias()}_num-epochs-{self.num_epochs}"
    if self.use_dog_aux_loss:
      model_alias += "_dog-aux-loss"
    saved_model_folder = os.path.join(data_folder, model_alias)
    if not os.path.exists(saved_model_folder):
      os.makedirs(saved_model_folder)

    # save model in the data alias folder
    saved_model_path = os.path.join(saved_model_folder, "model.pth")

    return saved_model_path

  def _train_loop(self, epoch: int) -> float:
    accum_loss = 0.0
    for i, batch in enumerate(self.train_data.data_loader):
      # flatten images and move to device
      images = batch[0].cuda()

      # forward pass and loss
      outputs = self.model(images)

      loss = nn.MSELoss()(outputs.view(outputs.size(0), -1), images.view(images.size(0), -1))

      outputs = outputs.view(*images.size())

      if self.use_dog_aux_loss:
        dog_loss = nn.MSELoss()(self.dog_filter(outputs), self.dog_filter(images))
        loss += (int(epoch * 0.3) * 2) * dog_loss

      accum_loss += loss.item()

      # backward pass and optimize
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # print and log loss
      self.writer.add_scalar('Loss/train', loss.item(), epoch * self.num_batches["train"] + i)

    return accum_loss / self.num_batches["train"]

  def _dev_loop(self, epoch: int, dev_data: ImageData, dev_data_name: str) -> float:
    # evaluate on given dev set
    accum_loss = 0.0
    with torch.no_grad():
      for i, batch in enumerate(dev_data.data_loader):
        # flatten images and move to device
        images = batch[0].cuda()

        # forward pass and loss
        outputs = self.model(images)
        loss = nn.MSELoss()(outputs.view(outputs.size(0), -1), images.view(images.size(0), -1))
        accum_loss += loss.item()

        # print and log loss
        self.writer.add_scalar(f'Loss/{dev_data_name}', loss.item(), epoch * self.num_batches[dev_data_name] + i)

        # save example reconstructions
        if i == 0 and epoch in self.sample_epochs:
          for j in range(self.num_samples):
            sample_input = batch[0][j].permute(1, 2, 0).cpu()
            sample_output = outputs[j].permute(1, 2, 0).cpu()
            if epoch == 0:
              self.sample_inputs[dev_data_name].append(sample_input)
            self.sample_outputs[dev_data_name][epoch].append(sample_output)

    return accum_loss / self.num_batches[dev_data_name]

  def _save_visualizations(self):
    # save example reconstructions
    num_sample_epochs = len(self.sample_epochs)
    for dev_data_name in ("dev", "devtrain"):
      fig, axarray = plt.subplots(
        self.num_samples, num_sample_epochs + 1, figsize=(num_sample_epochs + 1, self.num_samples))
      for i in range(self.num_samples):
        axarray[i, 0].imshow(self.sample_inputs[dev_data_name][i])
        for j, epoch in enumerate(self.sample_epochs):
          axarray[i, j + 1].imshow(self.sample_outputs[dev_data_name][epoch][i])
      # hide all ticks
      for ax in axarray.flatten():
        ax.tick_params(axis='both', which='both', length=0, labelsize=0)
      figure_path = os.path.join(self.visualizations_folder, f"{dev_data_name}_samples.png")
      plt.savefig(figure_path)
      plt.close(fig)
