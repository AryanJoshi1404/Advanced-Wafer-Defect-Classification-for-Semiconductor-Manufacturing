import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import resnet101, ResNet101_Weights
from tqdm import tqdm  # For progress bar
from data_preprocessing import get_data_loaders  # Import your data loader function
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths to your dataset
train_dir = "data//train"
val_dir = "data//validation"
test_dir = "data//test"

# Load data loaders
train_loader, val_loader, test_loader, class_names, num_classes = get_data_loaders(train_dir, val_dir, test_dir)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Check which device is being used

# Load the pretrained ResNet50 model
model = models.resnet101(weights=ResNet101_Weights.DEFAULT)

# Modify the final layer for your number of classes
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to the appropriate device
model.to(device)

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Load the model if resuming training
model_path = 'resnet101_model_epoch_last.pth'
resume_epoch = 0
if os.path.exists(model_path):  # Change to your latest saved model file name
    model.load_state_dict(torch.load(model_path))
    print("ResNet-101 Model loaded for resuming training.")
    resume_epoch = int(model_path.split('_')[-1].split('.')[0])  # Assumes format 'model_epoch_X.pth'
    print(f"Model loaded for resuming training from epoch {resume_epoch}.")


# Initialize variables for early stopping
best_val_loss = float('inf')
patience = 5
epochs_without_improvement = 0

# Track losses and accuracy for visualization
train_losses, val_losses = [], []
val_accuracies = []

# Training loop
num_epochs = 20  # Adjust as necessary
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')  # Save as the last model
        print(f"Model saved after epoch {epoch + 1}.")

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss=0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss=criterion(outputs,labels)
            val_loss+=loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and print average validation loss and accuracy for the epoch
    avg_val_loss = val_loss / len(val_loader)
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    # print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break  # Exit the training loop

    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')  # Save checkpoint every 2 epochs
        print(f"Model checkpoint saved after epoch {epoch + 1}.")


# Save the final model
torch.save(model.state_dict(), 'final_model.pth')
print("Final model saved as final_model.pth")

# Plotting the results
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.show()

all_preds=[]
all_labels=[]

# Testing loop
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        # Store predictions and true labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Test Accuracy: {100 * test_correct / test_total:.2f}%")

# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(classification_report(all_labels, all_preds, target_names=class_names))