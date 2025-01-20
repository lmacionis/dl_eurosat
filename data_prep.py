import os
from PIL import Image
import matplotlib.pyplot as plt


satelite_classes = []
path = "EuroSAT_dataset\\2750"

for folder in os.listdir(path):
    satelite_classes.append(folder)

print("Classes: ", satelite_classes, len(satelite_classes))

fig, axes = plt.subplots(1, 10, figsize=(15, 15))
for i, satelite_class in enumerate(satelite_classes):
    sample_image_path = os.path.join(path, satelite_class, os.listdir(os.path.join(path, satelite_class))[0])
    img = Image.open(sample_image_path)
    axes[i].imshow(img)
    #axes[i].set_title(satelite_classes)
    axes[i].axis("off")
plt.show()