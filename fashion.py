import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#fashion mnist 데이터 로드 28x28이미지 레이블0~9
fm = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fm.load_data()

c_names = ['T-shirt', 'Trouser', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape)
print(test_images.shape)

#픽셀값 범위 0~1로 조정
train_images = train_images / 255.0

test_images = test_images / 255.0


#28x28을 784 1차원으로 변환
#은닉층 128 노드 출력층 10노드
#완전연결
#10개 레이블 중 하나에 속할 확률 출력
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
#옵티마이저, 손실함수, 지표 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#훈련
model.fit(train_images, train_labels, epochs=5)

#테스트 셋으로 모델 정확도 측정
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)

#이미지 예측
predictions = model.predict(test_images)

#이미지 아래 예측 레이블과 정답 레이블 출력
#일치하면 초록 불일치 빨강
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(c_names[predicted_label], 100*np.max(predictions_array), c_names[true_label]), color=color)

#레이블에 속할 확률 바차트로 표시
#정답 바 초록, 오답 바 빨강, 나머지 회색
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')
    

#테스트 셋 처음 15개 출력
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

#테스트 셋으로 측정한 정확도는 훈련 중 측정된 정확도보다 낮음. 테스트 정확도 평균 0.87 정도로 측정됨
#바지나 셔츠, 아우터, 발목 부츠 등은 정확도가 높고 샌들과 운동화 등은 예측 정확도가 낮고 훈련 할 때마다 변동이 심함. 사진 해상도가 낮고 흑백이여서 큰 틀이 비슷한 것들은 서로 다른 특성이 많이 들어나지 않아서 그런 듯 힘.