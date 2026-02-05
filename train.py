import torch
from torch import nn, optim
from torchvision.models import resnet34
from torchvision import models
import data_setup
import pre_models


NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = len(data_setup.classes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = pre_models.create_resnet34_model(NUM_CLASSES)
model = model.to(device)

train_dataloader = data_setup.train_dataloader

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def accuracy_fn(outputs, labels):
  y_pred = torch.softmax(outputs, dim=1).argmax(1)
  correct = torch.eq(labels, y_pred).sum().item() / len(labels)
  acc = correct * 100
  return acc

print("Starting training...")
train_losses = []
train_accuracies = []

def train_step(dataloader, model, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        acc = accuracy_fn(y_logits, y)
        train_acc += acc

    avg_loss = train_loss / len(dataloader)
    accuracy = train_acc / len(dataloader)
    return avg_loss, accuracy


for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_step(train_dataloader, model, loss_fn, optimizer, device)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)


    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
torch.save(model.state_dict(), 'face_mask_detection_model.pth')
print('Model training complete and saved as face_mask_detection_model.pth')