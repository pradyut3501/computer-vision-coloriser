import os
import random
from skimage import io, img_as_ubyte
from skimage.color import rgba2rgb, rgb2gray
from tqdm import tqdm
import numpy as np
import pickle

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
            rgb_face = img_as_ubyte(rgb_face)
            gray_face = rgb2gray(rgb_face)
            gray_face = img_as_ubyte(gray_face)

            rgb_imgs.append(rgb_face)
            gray_imgs.append(gray_face)

    print(f"Identities loaded: {len(rgb_imgs)}")
    return rgb_imgs, gray_imgs


def preprocess():
    parts = ["part1", "part2", "part3"]
    all_rgb_imgs = []
    all_gray_imgs = []

    for i, part in enumerate(parts):
        print(f"Preprocessing part {i + 1}")
        processed_rgb, processed_gray = get_unique_ids(part)
        all_rgb_imgs += processed_rgb
        all_gray_imgs += processed_gray

    return np.array(all_rgb_imgs), np.array(all_gray_imgs)


class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self):
        self.color_imgs, self.gray_imgs = self.load_data()

        # TODO
        self.train_data = None
        self.test_data = None

    def store_data(self):
        all_rgb_imgs, all_gray_imgs = preprocess()
        file = open(DATA_DIR + "/pickled_data", "ab")
        pickle.dump(all_rgb_imgs, file)
        pickle.dump(all_gray_imgs, file)
        file.close()

    def load_data(self):
        file = open(DATA_DIR + "/pickled_data", "rb")
        all_rgb_imgs = pickle.load(file)
        all_rgb_imgs = all_rgb_imgs.astype(float)
        all_gray_imgs = pickle.load(file)
        all_gray_imgs = np.expand_dims(all_gray_imgs, axis=-1)
        all_gray_imgs = all_gray_imgs.astype(float)
        file.close()
        return all_rgb_imgs, all_gray_imgs
