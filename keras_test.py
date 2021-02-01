import sys

import numpy as np
import os
import tensorflow.compat.v1 as tf

# from autodist import AutoDist
# from autodist.strategy import PS
from tensorflow.python.training.training_util import get_or_create_global_step
import matplotlib.pyplot as plt

# autodist = AutoDist(
#     resource_spec_file='resource_spec.yml', 
#     strategy_builder=PS(local_proxy_variable=False, sync=True, staleness=3)
# )

# d = autodist

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[:, :, :, None]
test_images = test_images[:, :, :, None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

BATCH_SIZE = 128

EPOCHS = 1

#with tf.Graph().as_default(), d.scope():
x = tf.keras.Input(shape=(28, 28, 1))
y = tf.keras.Input(shape=())

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation='softmax')
])
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='auto')
# optimizer = tf.keras.optimizers.Adam()

# model.fit(train_images, train_labels, epochs=EPOCHS)
print(model.summary())
print(model.trainable_variables)

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)

# x = tf.placeholder(tf.float32, shape=[10,None], name="tmp")
# x = tf.reshape(x, shape=[tf.shape(x)[0],10])
# print(x)

x = tf.placeholder(tf.float32, shape=[None,10], name="tmp")
x = tf.reshape(x, shape=[5408,10])
print(x)