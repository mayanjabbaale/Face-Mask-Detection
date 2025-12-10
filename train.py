import matplotlib.pyplot as plt
import torch
from torchvision import models
from setup import train_dataloader, test_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights = models.ResNet34_Weights.DEFAULT
model = models.resnet34(weights=weights)
model = model.to(device)


for param in model.parameters():
    param.requires_grad = False


model.fc = torch.nn.Linear(in_features=512, out_features=3)
model.fc = model.fc.to(device)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def accuracy(y_pred, y_true):
    correct = (y_pred == y_true).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc


epochs = 15
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(epochs):
    model.train()
    train_loss, train_acc = 0, 0
    
    for data, labels in train_dataloader:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        y_logits = model(data)
        y_pred = torch.softmax(y_logits, dim=1).argmax(1)
        
        batch_loss = loss_fn(y_logits, labels)
        batch_acc = accuracy(y_pred=y_pred, y_true=labels)
        
        train_loss += batch_loss.item()
        train_acc += batch_acc
        
        batch_loss.backward()
        optimizer.step()
    
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    

    model.eval()
    test_loss, test_acc = 0, 0
    
    with torch.no_grad():
        for data, labels in test_dataloader:
            data, labels = data.to(device), labels.to(device)
            
            y_logits = model(data)
            y_pred = torch.softmax(y_logits, dim=1).argmax(1)
            
            batch_loss = loss_fn(y_logits, labels)
            batch_acc = accuracy(y_pred=y_pred, y_true=labels)
            
            test_loss += batch_loss.item()
            test_acc += batch_acc
    
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    scheduler.step()
    
    print(f'Epoch {epoch+1:02d}/{epochs}: '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
          f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(test_losses, label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Loss over epochs')

ax2.plot(train_accs, label='Train Acc')
ax2.plot(test_accs, label='Test Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.set_title('Accuracy over epochs')

plt.tight_layout()
plt.show()

torch.save(model.state_dict(), 'face_mask_model.pth')
print("Model saved as 'face_mask_model.pth'")