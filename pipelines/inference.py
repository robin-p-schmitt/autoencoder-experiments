import os
import torch
import matplotlib.pyplot as plt


def inference(data, model, saved_model_path):
  # load model from checkpoint
  if not os.path.exists(saved_model_path):
    print("Model not found, aborting inference.")
    return
  model.load_state_dict(torch.load(saved_model_path))

  # visualize some images
  with torch.no_grad():
    for batch in data.data_loader:
      images = batch[0].cuda()
      outputs = model(images)
      outputs = outputs.view(outputs.size(0), *data.img_size)
      for i in range(batch[0].size(0)):
        # show original image and reconstructed image next to each other
        fig, axarray = plt.subplots(1, 2)
        axarray[0].imshow(batch[0][i].permute(1, 2, 0).cpu())
        axarray[1].imshow(outputs[i].cpu().permute(1, 2, 0).cpu())
        plt.show()
