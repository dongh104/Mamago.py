!pip install tensorflow
!pip install opencv-python

//
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt

from google.colab import drive
drive.mount('/content/drive')
//

word_path = "/content/drive/MyDrive/Colab Notebooks/data/dataset"

def image_pretreatment(word):
  word_folder_path = os.path.join(word_path, word)
  if os.path.exists(word_folder_path):
    pass
  else:
    os.makedirs(word_folder_path)

  word_image_path = word_path + f"/{word}.PNG"
  image = cv2.imread(word_image_path)
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # 배경이 될 수 있는 색상 범위 설정
  lower_background = np.array([10], dtype = np.uint8)
  upper_background = np.array([240], dtype = np.uint8)

  mask = cv2.inRange(image_gray, lower_background, upper_background)

  # 텍스트와 배경을 분리
  text = cv2.bitwise_and(image_gray, image_gray, mask=mask)
  background = cv2.bitwise_and(image_gray, image_gray, mask=~mask)

  # 배경색 구분을 제대로 인식하지 못할 가능성을 생각하여 블러처리
  blur = cv2.GaussianBlur(background - text, ksize=(5,5), sigmaX=0)
  ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
  # 이미지의 윤곽선 검출
  edged = cv2.Canny(blur, 5, 5)

  # 텍스트 영역의 윤곽선 찾기
  contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

  # 각 윤곽선에 대한 경계 상자 그리기 및 알파벳 추출
  alphabet_images = []
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # 경계 상자 그리기
    cv2.rectangle(edged, (x, y), (x - w, y - h), (0, 0, 0), 1)

    # 알파벳 추출
    alphabet = edged[y:y+h, x:x+w]
    alphabet_images.append(alphabet)


  # 추출된 알파벳 별로 이미지 저장
  for i, alphabet in enumerate(alphabet_images):
    cv2.imwrite(f'{word_folder_path}/alphabet_{i}.png', alphabet)
//

def word_labeling(word):
    word_folder_path = os.path.join(word_path, word)
    labeled_data = []

    # 단어 폴더 내에서 "alphabet"이 들어간 이미지 파일을 찾기
    for filename in os.listdir(word_folder_path):
        if "alphabet" in filename:
            # 파일의 전체 경로
            file_path = os.path.join(word_folder_path, filename)

            # 이미지 읽기
            alphabet_image = cv2.imread(file_path)

            # 알파벳 이미지 표시
            cv2_imshow(alphabet_image)

            # 사용자에게 라벨 입력 받기
            while True:
                label = input(f"라벨링 될 알파벳을 입력하시오. : ")

                if label.lower() == '1':
                    break

                # Add label to the list
                labeled_data.append(label)
                break

    # 라벨 데이터 저장
    save_path = os.path.join(word_folder_path, f"{word}.txt")
    with open(save_path, 'w') as file:
        for data in labeled_data:
            file.write(f"{data}\n")

    print(f"{save_path}에 라벨링 지정이 완료 되었습니다.")
//

# 데이터 로드
word_path = "/content/drive/MyDrive/Colab Notebooks/data/dataset"
data = []
labels = []

for word in os.listdir(word_path):
    word_folder_path = os.path.join(word_path, word)

    # 라벨 파일 경로
    label_file_path = os.path.join(word_folder_path, f"{word}.txt")

    # 라벨 파일이 존재하는 경우에만 처리
    if os.path.exists(label_file_path):
        # 라벨 파일 읽기
        with open(label_file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]

        for filename, label in zip(os.listdir(word_folder_path), lines):
            if "alphabet" in filename:
                # 이미지 읽기
                img_path = os.path.join(word_folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))
                data.append(img)

                # 라벨 추가
                labels.append(label)

# 데이터 전처리
data = np.array(data) / 255.0
data = data.reshape((data.shape[0], 64, 64, 1))

# 라벨 인코딩
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# 데이터 분할
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 42)

# 모델 구성
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(len(set(labels)), activation = 'softmax'))

# 모델 컴파일
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# 모델 학습
model.fit(train_data, train_labels, epochs = 15, validation_data = (test_data, test_labels))
//
def Test(word):
  #image_pretreatment(word)
  test_image_path = word_path + "/" + word
  output = ""

  for filename in os.listdir(test_image_path):
    if "alphabet" in filename:
      path = os.path.join(test_image_path, filename)
      test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      test_image = cv2.resize(test_image, (64, 64))
      test_image = test_image.reshape((1, 64, 64, 1)) / 255.0

      predictions = model.predict(test_image)
      predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])
      output += predicted_label[0]

  print(output)
//
image_pretreatment("initialization")
//
word_labeling("initialization")
//
Test("experience")
//
