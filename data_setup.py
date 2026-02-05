import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import matplotlib.pyplot as plt

data_dir = Path('Dataset/data')
test_dir = Path('Dataset/eval')

BATCH_SIZE = 32


transforms = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=15),
])

train_data = ImageFolder(root=data_dir, transform=transforms)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = ImageFolder(root=test_dir, transform=transforms)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

classes = train_data.classes
img, label = next(iter(train_dataloader))

fig = plt.figure(figsize=(8, 8))
nrows, nclos = 3, 3

for i in range(1, nrows * nclos + 1):
    random_index = torch.randint(0, img.size(0), (1,)).item()
    fig.add_subplot(nrows, nclos, i)
    plt.imshow(img[random_index].permute(1, 2, 0))
    plt.title(classes[label[random_index]])
    plt.axis('off')

plt.show()