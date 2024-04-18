# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     keras
   Description :
   Author :       lizhenhui
   date：          2024/3/17
-------------------------------------------------
   Change Activity:
                   2024/3/17:
-------------------------------------------------
"""
import numpy as np
import mnist
# from keras.optimizers import SGD
# from tensorflow.python.keras.utils import to_categorical
# from tensorflow.python.keras.optimizers import SGD

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.utils.np_utils import to_categorical



train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

model = Sequential([
    Conv2D(8, 3, input_shape=(28, 28, 1), use_bias=False),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(10, activation='softmax'),
])

model.compile(SGD(learning_rate=.005), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_images,
    to_categorical(train_labels),
    batch_size=1,
    epochs=30,
    validation_data=(test_images, to_categorical(test_labels)),
)