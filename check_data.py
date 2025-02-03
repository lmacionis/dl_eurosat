import os
from PIL import Image
import matplotlib.pyplot as plt


satelite_classes = []
path = "EuroSAT_dataset/2750"

for folder in os.listdir(path):
    satelite_classes.append(folder)

print("Classes: ", satelite_classes, len(satelite_classes))


# Show example of data
fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i, satelite_class in enumerate(satelite_classes):
    sample_image_path = os.path.join(path, satelite_class, os.listdir(os.path.join(path, satelite_class))[0])
    img = Image.open(sample_image_path)
    axes[i].imshow(img)
    axes[i].set_title(satelite_class)
    axes[i].axis("off")
#plt.show()
plt.close()

# Count pictures in every class
number_dict = {}
for cls in satelite_classes:
    number_dict[cls] = len(os.listdir(path + "//" + cls))
    
print(number_dict)

# Show data in plot bar chart
plt.bar(number_dict.keys(), number_dict.values())
plt.xticks(rotation=45)
plt.ylabel("Image size")
plt.title("Number of images in each class")
#plt.show()
plt.close()
