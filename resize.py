# Importing required libraries
import os
from PIL import Image
import numpy as np

IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
IMAGE_DIR = './dataset256/'

all_images = []

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print('resizing...')

for filename in os.listdir(IMAGE_DIR):
    image_path = os.path.join(IMAGE_DIR, filename)
    image = Image.open(image_path)
    if image.width >= IMAGE_SIZE and image.height >= IMAGE_SIZE and np.asarray(image).shape[-1] == 3:
      all_images.append(image_path)

print('collected ' + str(len(all_images)) + ' images')

for path in all_images:
    Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS).save(path)

print('done')
