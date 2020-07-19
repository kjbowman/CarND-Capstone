import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

training_images_dir = r"/home/student/CarND-Capstone/training_images/"
red_images_dir = training_images_dir + r"0/"
yellow_images_dir = training_images_dir + r"1/"
green_images_dir = training_images_dir + r"2/"
training_images_dict = {
    red_images_dir: [1, 0, 0],
    yellow_images_dir: [0, 1, 0],
    green_images_dir: [0, 0, 1]
    }

def read_training_data():
    features = []
    labels = []
    n = 1
    for img_dir, label in training_images_dict.items():
        for filename in os.listdir(img_dir):
            print "\rreading file {1}: {0}".format(filename, n),
            img = cv2.imread(img_dir + filename)
            crop_img = img[50:-50, 150:-150]
            gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            cv2.normalize(gray_img, gray_img, -0.5, 0.5, cv2.NORM_MINMAX)
            features.append(np.array(gray_img, dtype=np.float32))
            labels.append(label)
            n += 1
    return np.array(features), np.array(labels)

x_train, y_train = read_training_data()
print "\nRead {0} training images".format(len(x_train))
x_train = np.reshape(x_train, x_train.shape+(1,))

permutation = np.random.permutation(x_train.shape[0])
x_train = x_train[permutation]
y_train = y_train[permutation]

# setup Keras
EPOCHS = 5
input_shape = (500, 500, 1)
n_filters = 32
k_size = (3, 3)
pooling_size = (2, 2)
drop_prob = 0.5
n_classes = 3

model = Sequential()
model.add(Conv2D(32, kernel_size=k_size, padding='valid',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pooling_size))
model.add(Conv2D(48, kernel_size=k_size, padding='valid',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pooling_size))
model.add(Dropout(drop_prob))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(drop_prob))
model.add(Dense(n_classes, activation='softmax'))

# preprocess data
# x_normalized = np.array(x_train / 255.0 - 0.5)

# compile and fit the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2)

print "done."