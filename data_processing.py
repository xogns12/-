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


# 숫자로 표현된 인종, 성별 데이터를 문자열로 변환
# 데이터셋을 사전에 설정
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


# 데이터셋을 파싱하여 정보를 추출하고 데이터프레임으로 변환하는 함수 정의
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


# 이미지 파일의 무작위 16개 이미지를 선택해 이미지에 나이, 성별, 인종 정보를 시각화
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
