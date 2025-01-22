# Import Packages
import pandas as pd
import matplotlib.pyplot  as plt
from pathlib import Path
import imagesize
from check_data import satelite_classes, path


# Get the Image Resolutions
# Get list of all images
all_img_path_list = []
for item in satelite_classes: 
    img_path_list = [img for img in Path(path + "\\" + item).iterdir() if img.suffix == ".jpg"]

    all_img_path_list.append(img_path_list)

# Get dict of all image resolutions
img_dict = {}

for dir in all_img_path_list:
    for file_name in dir:
        img_dict[str(file_name)] = imagesize.get(file_name)


# Create data frame in order to check how all images are scattered by size. 
img_dict_df = pd.DataFrame.from_dict([img_dict]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns')
img_dict_df[["Width", "Height"]] = pd.DataFrame(img_dict_df["Size"].tolist(), index=img_dict_df.index)

# Showing data in a Scatter Plot
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
points = ax.scatter(img_dict_df.Width, img_dict_df.Height, color='blue', alpha=0.5, label=f'Number of Images: {len(img_dict_df)}')
ax.set_title("Image Resolution")
ax.set_xlabel("Width", size=14)
ax.set_ylabel("Height", size=14)
ax.legend()
plt.show()
plt.close()
