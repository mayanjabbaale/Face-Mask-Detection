import numpy as np
from PIL import Image
import torchvision.transforms as T
import os
import torch
from pathlib import Path
import data_setup
import pre_models
import matplotlib.pyplot as plt

test_path = Path('Dataset/eval')

NUM_CLASSES = data_setup.classes
device = 'cuda' if torch.cuda.is_available() else 'cpu'


pred_transforms = T.Compose([
    T.ToTensor(),
])

test_images = []
not_in_test = []

for dir_path, dir_name, files in os.walk(test_path):
  for file in files:
    if file.endswith(".png"):
      test_images.append(os.path.join(dir_path, file))
    else:
      not_in_test.append(file)

pretrained_model = pre_models.create_resnet34_model(NUM_CLASSES)
pretrained_model.load_state_dict(state_dict=torch.load('face_mask-resnet34.pth'))

fig = plt.figure(figsize=(12, 13))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(1, len(test_images), size=[1]).item()
    img_path = test_images[random_idx]
    img = Image.open(img_path)

    # Get the actual class from the image path
    actual_class = Path(img_path).parent.stem

    logits = pretrained_model(pred_transforms(img).unsqueeze(0).to(device))
    pred_class = data_setup.classes[torch.softmax(logits, 1).argmax(1).item()]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img)
    plt.title(f'Actual: {actual_class}\nPredicted: {pred_class}')
    plt.axis(False)