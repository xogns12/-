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

# 특정 폴더 내 이미지 파일에서 나이, 성별 정보를 추출
# 이미지를 일정 크기로 변환한 후 배열로 저장
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


# 이미지를 나이, 성별 예측을 위한 훈련set, 검증set, test set으로 나누는 작업
x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42, test_size=0.4)
x_valid_age, x_test_age, y_valid_age, y_test_age = train_test_split(x_test_age, y_test_age, random_state=42, test_size = 0.5)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, gender, random_state=42, test_size=0.4)
x_valid_gender, x_test_gender, y_valid_gender, y_test_gender = train_test_split(x_test_gender, y_test_gender, random_state=42, test_size=0.5)


# 예측 모델을 위한 디렉토리 생성, 학습 관련 파라미터 및 콜백 설정
model_types = ['age_model', 'gender_model']

checkpoint_dirs = {model_type: f'./{model_type}' for model_type in model_types}
for dir in checkpoint_dirs.values():
    os.makedirs(dir, exist_ok=True)

init_lr = 1e-4
epochs = 30
opt = Adam(learning_rate=init_lr)

callbacks = {}
for model_type in model_types:
    callbacks[model_type] = [
        EarlyStopping(monitor='val_loss', patience=7),
        ModelCheckpoint(filepath=os.path.join(
            checkpoint_dirs[model_type],
            'model-{epoch:02d}-{val_loss:.2f}.h5'),
             monitor='val_loss',
             save_best_only=True)]
