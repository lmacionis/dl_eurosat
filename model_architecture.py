from check_data import path, satelite_classes
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# Transformacijos: normalizavimas
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.34, 0.38, 0.4], std=[0.091, 0.065, 0.055])
])

# Sukuriame PyTorch dataset

full_dataset = datasets.ImageFolder(path, transform=transform)

# Padaliname į treniravimo, validacijos ir testavimo rinkinius
train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Sukuriame duomenų įkėlimą (DataLoader)
"""Can try to change the batch_size from the data
https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network"""

train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=30, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

print(f"Training data: {len(train_dataset)} images")
print(f"Validation data: {len(val_dataset)} images")
print(f"Testing data: {len(test_dataset)} images")


class EurSatCNN(nn.Module):
    def __init__(self):
        super(EurSatCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # RGB 3 Channels, feature counting according to formula, kernel / filter priklauso nuo paveiksliuko dydžio jei mažas imam maža kernel, jei didelis t.y. dideli.
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.dropout = nn.Dropout(0.1)     # Can increace this according if its underfitting or overfitting
        self.pool = nn.MaxPool2d(2, 2)      # Naudojama sumažinti computational load, bei overfitting. (kernel size, stride). MaxPool naudojamas dažniausiai palyginus su avg ir adptive. 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096, 64)    # Crucial component used to transform the output of a CNN or a RNN into a format that can be used for classification or regression.
        self.fc2 = nn.Linear(64, len(satelite_classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))  # Fully connected sluoksnis
        x = self.fc2(x)  # Klasifikavimo sluoksnis
        return x

model = EurSatCNN()
print(model)

# # Santrauka su sluoksnių išvesties formomis ir parametrų skaičiumi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

summary(model, (3, 64, 64), device=device.type) 
