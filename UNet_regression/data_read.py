## 필요한 패키지 등록
import os
import numpy as np # !conda install numpy
from PIL import Image # !conda install pillow
import matplotlib.pyplot as plt # !conda install matplotlib

## 데이터 불러오기
# -- 2012 EM segmentation challenge
dir_data = './datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nfram = img_label.n_frames

## train/val/test directory 만들기
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## random하게 train/val/test index 만들기
id_frame = np.arrange(nframe)
np.random.shuffle(id_frame)

## training set을 만드는 구문
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe]) # 기준점 이동
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

## validation set을 만드는 구문
offset_nframe += nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## test set을 만드는 구문
offset_nframe += nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## 출력
plt.subplot(123)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(123)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()