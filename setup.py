from collections import Counter
import os
from pathlib import Path
from matplotlib import pyplot as plt
import xmltodict
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np

data = Path(r'C:\Users\p\OneDrive\Desktop\CoVision\Datasets\face-mask-data')
annotations = data/'annotations'
images = data/'images'

# Get image and XML files
image_names = []
xml_names = []

for dir_path, dir_name, file_names in os.walk(data):
    for file_name in file_names:
        file_path = os.path.join(dir_path, file_name)
        if file_path.endswith(('.jpg', '.jpeg', '.png')):
            image_names.append(file_name)
        elif file_path.endswith('.xml'):
            xml_names.append(file_name)

# Analyze dataset distribution
listing = []
for i in image_names:
    xml_path = annotations / (Path(i).stem + ".xml")
    if xml_path.exists():
        with open(xml_path) as fd:
            doc = xmltodict.parse(fd.read())
        temp = doc['annotation']['object']
        
        if isinstance(temp, list):
            for obj in temp:
                listing.append(obj["name"])
        else:
            listing.append(temp["name"])

items = list(Counter(listing).keys())
values = list(Counter(listing).values())

# Visualization (commented out in your original code)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
background_color = '#faf9f4'
ax1.set_facecolor(background_color)
ax2.set_facecolor(background_color)
ax1.pie(values, wedgeprops=dict(width=0.3, edgecolor='w'), labels=items, radius=1, startangle=120, autopct='%1.2f%%')
ax2.bar(items, values, color='maroon', width=0.4)
plt.title("Dataset Distribution")
plt.tight_layout()
plt.show()

options = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}

my_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),  # Changed to 224x224 for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def dataset_creation(image_list):
    image_tensor = []
    label_tensor = []
    
    for img_name in image_list:
        xml_path = annotations / (Path(img_name).stem + ".xml")
        if not xml_path.exists():
            continue
            
        with open(xml_path) as fd:
            doc = xmltodict.parse(fd.read())
        
        img_path = images / img_name
        if not img_path.exists():
            continue
            
        objects = doc["annotation"]["object"]
        if not isinstance(objects, list):
            objects = [objects]
        
        for obj in objects:
            x, y, w, h = list(map(int, obj["bndbox"].values()))
            label = options[obj["name"]]
            
            # Ensure bounding box is valid
            width = max(1, w - x)
            height = max(1, h - y)
            
            try:
                image = Image.open(img_path).convert("RGB")
                # Crop and transform
                cropped_img = transforms.functional.crop(image, y, x, height, width)
                transformed_img = my_transform(cropped_img)
                
                image_tensor.append(transformed_img)
                label_tensor.append(torch.tensor(label, dtype=torch.long))
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
    
    # Create dataset as list of (image, label) tuples
    final_dataset = list(zip(image_tensor, label_tensor))
    return final_dataset

# Create dataset
dataset = dataset_creation(image_names)
print(f"Total samples in dataset: {len(dataset)}")

# Split dataset
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size

train_data, test_data = torch.utils.data.random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

train_dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True, num_workers=0
)
test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=32, shuffle=False, num_workers=0
)

# Visualize batch
imgs, labels = next(iter(train_dataloader))

fig = plt.figure(figsize=(15, 5))
for idx in range(min(6, len(imgs))):
    ax = fig.add_subplot(2, 3, idx+1)
    # Denormalize for visualization
    img = imgs[idx].permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(f"Label: {labels[idx].item()}")
    ax.axis('off')
plt.tight_layout()
plt.show()