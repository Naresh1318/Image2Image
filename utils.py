import os
import sys
import numpy as np
from PIL import Image


def generate_input(image_path):
    image_paths = os.listdir(image_path)
    input_images = []
    target_images = []

    for i, img_path in enumerate(image_paths):
        img_path = image_path + '/' + img_path
        img = Image.open(img_path).resize([512, 256])
        img = np.array(img)
        img_height, img_width = img.shape[0], img.shape[1]
        input_img = img[:, :int(img_width/2), :]
        target_img = img[:, int(img_width/2):, :]
        input_images.append(input_img)
        target_images.append(target_img)
        print("\rLoading image: {}/{}".format(i, len(image_paths)), end="")
        sys.stdout.flush()

    input_images = np.array(input_images).reshape([-1, img_height, int(img_width/2), 3])/255.
    target_images = np.array(target_images).reshape([-1, img_height, int(img_width/2), 3])/255.

    return input_images, target_images
