from google.colab import drive
drive.mount('/content/drive')

!pip install kaggle
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d jangedoo/utkface-new
!unzip utkface-new.zip


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import seaborn as sns
import plotly.graph_objects as go
from glob import glob
import random

dataset_dict = {
    'race_id': {
        0: 'white',
        1: 'black',
        2: 'asian',
        3: 'indian',
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}
dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((g, i) for i, g in dataset_dict['race_id'].items())
folder_name = 'UTKFace'

def parse_dataset(dataset_path, ext='jpg'):
    def parse_info_from_file(path):
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')
            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None
    files = glob(os.path.join(dataset_path, "*.%s" % ext))
    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)

    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()
    return df

 df = parse_dataset(folder_name)
 df.head()

gender_dict = {'0': 'Male', '1': 'Female'}
race_dict = {'0': 'White', '1': 'Black', '2': 'Asian', '3': 'Indian', '4': 'Others'}
image_dir = '/content/UTKFace'
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')]
random_images = random.sample(image_files, 16)
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, image_path in enumerate(random_images):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    file_name = os.path.basename(image_path)
    age, gender, race = file_name.split('_')[:3]
    gender_text = gender_dict[gender]
    race_text = race_dict[race]
    ax = axes[i // 4, i % 4]
    ax.imshow(img_rgb)
    ax.set_title(f'Age: {age}\nGender: {gender_text}\nRace: {race_text}', fontsize=8)

plt.tight_layout()
plt.show()


import plotly.graph_objects as go
import plotly.express as px

def plot_distribution(pd_series):
    labels = pd_series.value_counts().index.tolist()
    counts = pd_series.value_counts().values.tolist()
    pie_plot = go.Pie(labels=labels, values=counts, hole=.3)
    fig = go.Figure(data=[pie_plot])
    fig.update_layout(title_text='Distribution for %s' % pd_series.name)
    fig.show()

plot_distribution(df['race'])
plot_distribution(df['gender'])

fig = px.histogram(df, x="age", nbins=20)
fig.update_layout(title_text='Age distribution')
fig.show()

bins = [0, 10, 20, 30, 40, 60, 80, np.inf]
names = ['<10', '10-20', '20-30', '30-40', '40-60', '60-80', '80+']
age_binned = pd.cut(df['age'], bins, labels=names)
plot_distribution(age_binned)


import os
import cv2
import numpy as np
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, Activation, Dropout, Lambda, Dense, Flatten, Input
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as k
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42, test_size=0.4)
x_valid_age, x_test_age, y_valid_age, y_test_age = train_test_split(x_test_age, y_test_age, random_state=42, test_size = 0.5)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, gender, random_state=42, test_size=0.4)
x_valid_gender, x_test_gender, y_valid_gender, y_test_gender = train_test_split(x_test_gender, y_test_gender, random_state=42, test_size=0.5)

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


age_model = Sequential([
    Conv2D(32, kernel_size=3, input_shape=(200,200,3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, strides=2),
    Conv2D(64, kernel_size=3),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, strides=2),
    Conv2D(128, kernel_size=3),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, strides=2),
    Dropout(0.25),
    Conv2D(256, kernel_size=3),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, strides=2),
    Dropout(0.25),
    Conv2D(512, kernel_size=3),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, strides=2),
    Flatten(),
    Dropout(0.25),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(1, activation='linear', name='age')])
opt = Adam()
age_model.compile(loss="mse", optimizer=opt, metrics=['mae'])
age_model.summary()

#age_model 학습
for model_type in model_types: 'age_model'
history_age = age_model.fit(x_train_age, y_train_age,
                        validation_data=(x_valid_age, y_valid_age),
                            batch_size = 32,
                            epochs=30,
                            callbacks = callbacks['age_model'])

predictions = age_model.predict(x_test_age)
mae = mean_absolute_error(y_test_age, predictions)
print(f"평균 절대 오차: {mae}")


#gender model
gender_model2 = Sequential([
    Conv2D(64, kernel_size=3, input_shape=(200,200,3)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, kernel_size=3),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, strides=2),
    Conv2D(128, kernel_size=3),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, strides=2),
    Conv2D(128, kernel_size=3),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, strides=2),
    Dropout(0.25),
    Conv2D(256, kernel_size=3),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, strides=2),
    Dropout(0.25),
    Conv2D(512, kernel_size=3),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=3, strides=2),
    Flatten(),
    Dropout(0.25),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid', name='gender')])
gender_model2.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
gender_model2.summary()


opt = Adam()
gender_model2.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
all_variables_age = age_model.trainable_variables
all_variables_gender = gender_model2.trainable_variables
all_variables = all_variables_age + all_variables_gender
opt.build(all_variables)

for model_type in model_types: 'gender_model'
history_gender = gender_model2.fit(x_train_gender, y_train_gender,
                        validation_data=(x_valid_gender, y_valid_gender),
                                  batch_size=64,
                                  epochs=30,
                                  callbacks = callbacks['gender_model'])

predictions = gender_model2.predict(x_test_gender)
predictions = [1 if prediction >= 0.5 else 0 for prediction in predictions]
accuracy = accuracy_score(y_test_gender, predictions)
print(f"정확도: {accuracy}")


plt.plot(history_age.history['loss'], label='Age Training Loss')
plt.plot(history_age.history['val_loss'], label='Age Validation Loss')
plt.title('Age Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history_gender.history['loss'], label='Gender Training Loss')
plt.plot(history_gender.history['val_loss'], label='Gender Validation Loss')
plt.title('Gender Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


age_predictions = age_model.predict(x_test_age)
age_predictions_rounded = np.round(age_predictions).astype(int)
unique_labels = np.unique(y_test_age)

print("Age Model Classification Report")
print(classification_report(y_test_age, age_predictions_rounded))

gender_predictions = gender_model2.predict(x_test_gender)
gender_predictions_binary = (gender_predictions > 0.5).astype(int)
print("Gender Model Classification Report")
print(classification_report(y_test_gender, gender_predictions_binary))


# 연령 모델의 혼동 행렬 계산
conf_matrix_age = confusion_matrix(y_test_age, age_predictions_rounded)
plt.figure(figsize=(16, 14))
sns.heatmap(conf_matrix_age, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - age Model')
plt.show()


# 성별 모델의 혼동 행렬 계산
conf_matrix_gender = confusion_matrix(y_test_gender, gender_predictions_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_gender, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Gender Model')
plt.show()


random_indices = np.random.choice(len(x_test_age), size=16, replace=False)
plt.figure(figsize=(12, 12))
for i, idx in enumerate(random_indices):
    image = x_test_age[idx]
    actual_age = int(y_test_age[idx])
    actual_gender = "Male" if y_test_gender[idx] == 1 else "Female"
    age_prediction = int(age_model.predict(np.expand_dims(image, axis=0))[0][0])
    gender_prediction = gender_model2.predict(np.expand_dims(image, axis=0))[0][0]
    predicted_gender = "Male" if gender_prediction >= 0.5 else "Female"
    plt.subplot(4, 4, i + 1)
    plt.imshow(image)
    plt.axis('off')
    plt.text(0, image.shape[0] + 10, f'Actual Age: {actual_age}, Gender: {actual_gender}',
             color='black', fontsize=10, ha='left') # 실제 나이와 성별 표시(검은색 글씨)
    plt.text(0, image.shape[0] + 30, f'Predicted Age: {age_prediction}, Gender: {predicted_gender}',
             color='red', fontsize=10, ha='left') # 예측된 나이와 성별 표시(빨간색 글씨)
plt.tight_layout()
plt.show()


from keras.models import load_model

#age_model 저장 후 불러오기
age_model.save('age_model.keras')
age_model = load_model('age_model.keras')

#gender_model 저장 후 불러오기
gender_model2.save('gender_model.keras')
gender_model = load_model('gender_model.keras')


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

age_model_path = "/content/age_model.keras"
gender_model_path = '/content/gender_model.keras'
age_model = load_model(age_model_path)
gender_model = load_model(gender_model_path)


# 얼굴 검출을 위한 Haar Cascade Classifier 로드
# haarcascade_frontalface_default.xml파일 다운로드 받는 페이지
# https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier("/content/drive/MyDrive/haarcascade_frontalface_default.xml")
image_size = 200
pic = cv2.imread('이미지_파일_경로.jpg')
faces = face_cascade.detectMultiScale(pic, scaleFactor=1.11, minNeighbors=8)

age_ = []
gender_ = []

for (x, y, w, h) in faces:
    img = pic[y:y + h, x:x + w]
    img = cv2.resize(img, (image_size, image_size))
    age_predict = age_model.predict(np.array(img).reshape(-1, image_size, image_size, 3))
    gender_predict = gender_model.predict(np.array(img).reshape(-1, image_size, image_size, 3))

    if len(gender_predict) > 0:
        age_.append(age_predict)
        gender_.append(np.round(gender_predict))
        gend = np.round(gender_predict)
        if gend == 0:
            gend = 'Man'
            col = (255, 255, 0)
        else:
            gend = 'Woman'
            col = (203, 12, 255)
        cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 225, 0), 1)
        cv2.putText(pic, "Age:" + str(int(age_predict)) + " / " + str(gend), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, w * 0.005, col, 1)
pic1 = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20, 16))
if len(faces) > 0:
    print(age_, gender_predict)
    plt.imshow(pic1)
    plt.show()
else:
    print("No faces detected.")
