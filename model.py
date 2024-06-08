import os
import cv2
import numpy as np
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, Activation, Dropout, Lambda, Dense, Flatten, Input
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as k


image_paths = []
ages = []
genders = []
for img_file in os.listdir(folder_name):
    img_path = os.path.join(folder_name, img_file)
    image_paths.append(img_path)
    age, gender = img_file.split("_")[:2]
    ages.append(int(age))
    genders.append(int(gender))

age = np.array(ages, dtype=np.int64)
gender = np.array(genders, dtype=np.uint64)

images = []
for img_path in image_paths:
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (200, 200))
    images.append(img_resized)
images = np.array(images)
