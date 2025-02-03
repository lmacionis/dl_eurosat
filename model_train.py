import torch.nn as nn
import torch
from tqdm import tqdm
from model_architecture import model, train_loader, val_loader

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=30):
    # Aptinkame GPU arba CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Perkeliame modelį į pasirinktą įrenginį
    model = model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        #Treniruojame
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epochs {epoch + 1}/{epoch}"):
            # Perkeliame duomenis į GPU arba CPU
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            #Apskaičiuojam, kaip turi būti atnaujinti svoriai pagal gautą loss
            loss.backward()

            #Atnaujinam svorius
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        # Validacija
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                # Perkeliame duomenis į GPU arba CPU
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)

        print(f"Epochs {epoch + 1}/{epochs}")
        print(f"   Train loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"   Validation loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer)