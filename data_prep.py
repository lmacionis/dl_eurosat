import os
from PIL import Image
import matplotlib.pyplot as plt

"""Export data into csv."""

satelite_classes = []
path = "EuroSAT_dataset\\2750"

for folder in os.listdir(path):
    satelite_classes.append(folder)

print("Classes: ", satelite_classes, len(satelite_classes))


# Show example of data
fig, axes = plt.subplots(1, 10, figsize=(20, 20))
for i, satelite_class in enumerate(satelite_classes):
    sample_image_path = os.path.join(path, satelite_class, os.listdir(os.path.join(path, satelite_class))[0])
    img = Image.open(sample_image_path)
    axes[i].imshow(img)
    axes[i].set_title(satelite_class)
    axes[i].axis("off")
#plt.show()


# Count pictures in every class
number_dict = {}
for dir in satelite_classes:
    lst = os.listdir(path + "//" + dir)
    number_files = len(lst)
    number_dict = {dir: number_files}
    print(number_dict)
