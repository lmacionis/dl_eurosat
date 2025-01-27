from check_data import path, satelite_classes
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


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

train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)

print(f"Treniravimo rinkinys: {len(train_dataset)} vaizdų")
print(f"Validacijos rinkinys: {len(val_dataset)} vaizdų")
print(f"Testavimo rinkinys: {len(test_dataset)} vaizdų")


class EurSatCNN(nn.Module):
    def __init__(self):
        super(EurSatCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, 2, stride=1, padding=1, padding_mode="reflect")   # RGB 3 Channels, feature countint according to formula, kernel / filter priklauso nuo paveiksliuko dydžio jei mažas imam maža kernel, jei didelis t.y. dideli.
        self.dropout1 = nn.Dropout(0.2)     # Can increace this according if its underfitting or overfitting
        self.conv2 = nn.Conv2d(2, 4, 2, stride=1, padding=1, padding_mode="reflect")
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(4, 8, 2, stride=1, padding=1, padding_mode="reflect")
        self.dropout3 = nn.Dropout(0.2)
        self.conv4 = nn.Conv2d(8, 16, 2, stride=1, padding=1, padding_mode="reflect")
        self.dropout4 = nn.Dropout(0.2)
        self.conv5 = nn.Conv2d(16, 32, 2, stride=1, padding=1, padding_mode="reflect")
        self.pool = nn.MaxPool2d(2, 2)      # Naudojama sumažinti computational load, bei overfitting. (kernel size, stride). MaxPool naudojamas dažniausiai palyginus su avg ir adptive. 
        self.fc1 = nn.Linear(32 * 16 * 8 * 4 * 2 * 2, 64)   # Crucial component used to transform the output of a CNN or a RNN into a format that can be used for classification or regression.
        self.fc2 = nn.Linear(128, len(satelite_classes))
