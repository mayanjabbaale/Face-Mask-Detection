from pathlib import Path
import torch
import torchvision
from torchvision.models import resnet34
import PIL.Image as Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

state_dict = Path('./Face-Mask-Detection/face_mask_model.pth')
model = resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
model.fc = torch.nn.Linear(in_features=512, out_features=3)

for param in model.parameters():
    param.requires_grad = False


model.load_state_dict(torch.load(state_dict, map_location=device))
model.to(device)

my_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

face_detector = MTCNN(keep_all=True, device=device)
yolo_face_detector = YOLO("./Face-Mask-Detection/yolov8n-face.pt")

def prediction(image):
    model.eval()
    img = Image.open(image).convert("RGB")
    boxes, _ = face_detector.detect(img)

    if boxes is None:
        return []
    
    results = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = img.crop((x1, y1, x2, y2))

        face_transform = my_transform(face).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(face_transform)
            probs = torch.softmax(logits, dim=1)[0]
            prob = probs[probs.argmax().item()]
            pred_class = classes[probs.argmax().item()]

            results.append({
                "box": (x1, y1, x2, y2),
                "class": pred_class,
                "prob": prob
            })
    
    return results

def yolo_prediction(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    results = yolo_face_detector(img_np)[0]  # first image result

    faces = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Crop face from the original image
        face = img.crop((x1, y1, x2, y2))

        # Transform for mask classifier
        face_t = my_transform(face).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(face_t)
            probs = torch.softmax(logits, dim=1)[0]

        pred_idx = probs.argmax().item()
        pred_class = classes[pred_idx]
        pred_prob = float(probs[pred_idx].item())

        faces.append({
            "box": (x1, y1, x2, y2),
            "class": pred_class,
            "prob": pred_prob
        })

    return faces


class_colors = {
    "with_mask": "green",
    "without_mask": "red",
    "mask_weared_incorrect": "yellow"
}

def show_predictions(image_path, results):
    img = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(1, figsize=(10, 10), dpi=150)
    ax.imshow(img, interpolation='lanczos')
    ax.axis("off")

    for r in results:
        x1, y1, x2, y2 = r["box"]
        label = r["class"]
        prob = r["prob"]
        color = class_colors[label]  # get color based on class

        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        # Draw label
        ax.text(
            x1,
            y1 - 4,
            f"{label} ({prob:.2f})",
            color="black",
            fontsize=5,
            weight="bold",
            bbox=dict(facecolor=color, alpha=0.5, edgecolor="none")
        )

    plt.tight_layout()
    plt.show()


image_path = state_dict = Path('./Face-Mask-Detection/examples/image_5.png')
results = prediction(image_path)
show_predictions(image_path, results)

