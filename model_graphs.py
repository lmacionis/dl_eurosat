from model_train import train_losses, train_accuracies, val_losses, val_accuracies
import matplotlib.pyplot as plt 
from model_architecture import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

# Grafikai
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss every epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training accuracy")
plt.plot(val_accuracies, label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy change per epoch")
plt.legend()
plt.tight_layout()
plt.show()

# Test model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Perkeliame duomenis į GPU arba CPU
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Testing accuracy: {100 * correct / total:.2f}%")

test_model(model, test_loader)


# Generate the confusion matrix
# Get predictions for the test dataset
y_true = []
y_pred = []

# Disable gradient calculation during inference
with torch.no_grad():
    for images, labels in test_loader:
        # Move data to the device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Get model predictions
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Append true and predicted labels to the lists
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Get class names from the dataset
classes = test_dataset.dataset.classes

cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
# Setting figsize
fig, ax = plt.subplots(figsize=(15, 15))
disp.plot(cmap=plt.cm.Blues, ax=ax) # Plot on the created axes

plt.title('Confusion Matrix', fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
