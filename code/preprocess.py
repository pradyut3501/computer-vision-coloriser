import os
import random
from skimage import io, img_as_float32
from skimage.color import rgba2rgb, rgb2gray
from tqdm import tqdm
import numpy as np
import pickle
import math

DATA_DIR = "../data"


def get_unique_ids(part):
    id_folders = os.listdir(DATA_DIR + "/raw/" + part)
    rgb_imgs = []
    gray_imgs = []

    for id in tqdm(id_folders, total=len(id_folders)):
        # Ignore hidden files
        if not id.startswith("."):
            all_faces = os.listdir(DATA_DIR + "/raw/" + part + "/" + id)
            # Randomly choose image for identity
            chosen_face = random.choice(all_faces)
            chosen_face = io.imread(DATA_DIR + "/raw/" + part + "/" +
                                    id + "/" + chosen_face)
            # Convert image from RGBA to RGB and grayscale
            rgb_face = rgba2rgb(chosen_face)
            rgb_face = img_as_float32(rgb_face)
            gray_face = rgb2gray(rgb_face)
            gray_face = img_as_float32(gray_face)
            gray_face = np.expand_dims(gray_face, axis=-1)

            rgb_imgs.append(rgb_face)
            gray_imgs.append(gray_face)

    print(f"Identities loaded: {len(rgb_imgs)}")
    return np.array(rgb_imgs), np.array(gray_imgs)


def preprocess():
    parts = ["part1", "part2", "part3"]

    for i, part in enumerate(parts):
        print(f"Preprocessing part {i + 1}")
        rgb_imgs, gray_imgs = get_unique_ids(part)

        # Train/validation/test split: 80-10-10
        idxs = np.random.permutation(len(rgb_imgs))
        train_i = math.floor(0.8 * len(rgb_imgs))
        val_i = math.floor(0.9 * len(rgb_imgs))

        train_idxs = idxs[:train_i]
        val_idxs = idxs[train_i:val_i]
        test_idxs = idxs[val_i:]

        # Store in file for train images
        train_rgb_imgs = rgb_imgs[train_idxs]
        train_gray_imgs = gray_imgs[train_idxs]
        print("Storing in file for train RGB images")
        store_data("train_rgb", train_rgb_imgs)
        print("Storing in file for train gray images")
        store_data("train_gray", train_gray_imgs)
        # Store in file for validation images
        val_rgb_imgs = rgb_imgs[val_idxs]
        val_gray_imgs = gray_imgs[val_idxs]
        print("Storing in file for validation RGB images")
        store_data("validation_rgb", val_rgb_imgs)
        print("Storing in file for validation gray images")
        store_data("validation_gray", val_gray_imgs)
        # Store in file for test images
        test_rgb_imgs = rgb_imgs[test_idxs]
        test_gray_imgs = gray_imgs[test_idxs]
        print("Storing in file for test RGB images")
        store_data("test_rgb", test_rgb_imgs)
        print("Storing in file for test gray images")
        store_data("test_gray", test_gray_imgs)


def store_data(file_name, imgs):
    file = open(DATA_DIR + "/pickled/" + file_name, "ab")
    pickle.dump(imgs, file)
    file.close()


def load_data(file_name):
    imgs = []
    file = open(DATA_DIR + "/pickled/" + file_name, "rb")
    while True:
        try:
            imgs.append(pickle.load(file))
        except EOFError:
            file.close()
            return np.concatenate(imgs)


class Datasets():
    """ Class for containing the training, validation, and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, is_stored=True):
        if not is_stored:
            preprocess()

        # Load train images
        self.train_rgb_imgs = load_data("train_rgb")
        self.train_gray_imgs = load_data("train_gray")
        # Load validation images
        self.val_rgb_imgs = load_data("validation_rgb")
        self.val_gray_imgs = load_data("validation_gray")
        # Load test images
        self.test_rgb_imgs = load_data("test_rgb")
        self.test_gray_imgs = load_data("test_gray")


def main():
    data = Datasets()
    print(f"Train RGB: {data.train_rgb_imgs.shape}")
    print(f"Train gray: {data.train_gray_imgs.shape}")
    print(f"Val RGB: {data.val_rgb_imgs.shape}")
    print(f"Val gray: {data.val_gray_imgs.shape}")
    print(f"Test RGB: {data.test_rgb_imgs.shape}")
    print(f"Test gray: {data.test_gray_imgs.shape}")


if __name__ == '__main__':
    main()
