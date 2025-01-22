from check_data import *
import torch
from torchvision import datasets, transforms


N_CHANNELS = 3  # According to RGB 3 channels
dataset = datasets.ImageFolder(path, transform=transforms.ToTensor())
full_loader = torch.utils.data.DataLoader(dataset, shuffle=False)

mean = torch.zeros(N_CHANNELS)
std = torch.zeros(N_CHANNELS)
print('==> Computing mean and std..')
for inputs, _labels in full_loader:
    for i in range(N_CHANNELS):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()
# Normalize the mean and std by deviding of the total number of images in the dataset.
mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)
