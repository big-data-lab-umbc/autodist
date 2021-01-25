import os
import sys
import numpy as np
import tensorflow as tf
from autodist.strategy import PS
#from keras import backend as K
import matplotlib.pyplot as plt

############################################################
# Step 1: Construct AutoDist with ResourceSpec
from autodist import AutoDist
autodist = AutoDist(
    resource_spec_file='resource_spec.yml', 
    strategy_builder=PS(local_proxy_variable=False, sync=True, staleness=3)
)
############################################################

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images[:, :, :, None]
test_images = test_images[:, :, :, None]
train_labels = train_labels[:]
test_labels = test_labels[:]
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

BATCH_SIZE = 64
EPOCHS = 1

#############################################################
# Step 2: Build with Graph mode, and put it under AutoDist scope
with tf.Graph().as_default(), autodist.scope():
#############################################################

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).repeat(EPOCHS).shuffle(len(train_images)//2).batch(BATCH_SIZE)

    # test_dataset = tf.data.Dataset.from_tensor_slices(
    #     (test_images, test_labels)).repeat(EPOCHS).batch(BATCH_SIZE)
    train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # model = tf.keras.Sequential()
    # # Must define the input shape in the first layer of the neural network
    # model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(10, activation='softmax'))    
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD()
    #train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def train_step(inputs):
        x, y = inputs
        y_hat = model(x, training=True)
        loss = loss_fn(y, y_hat)
        all_vars = []
        for v in model.trainable_variables:
            all_vars.append(v)
        grads = tf.gradients(loss, all_vars)
        update = optimizer.apply_gradients(zip(grads, all_vars))

        return loss

    fetches = train_step(train_iterator)
    #############################################################
    # Step 3: create distributed session
    sess = autodist.create_distributed_session()
    #############################################################
    iteration = len(train_images) // BATCH_SIZE * EPOCHS
    losses = []
    for _ in range(iteration):
        #train_accuracy.reset_states()
        loss = sess.run(fetches)
        losses.append(loss)
        print(f"train_loss: {loss}")
        #print(f'train_accuracy: {train_accuracy.result().eval() * 100}')

    plt.plot(range(iteration-2), losses[2:], 'bo', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./mygraph.png")

    sess.close()    

