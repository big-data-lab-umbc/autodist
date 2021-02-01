import sys
import numpy as np
import os
import tensorflow.compat.v1 as tf

from autodist import AutoDist
from autodist.strategy import PS
import matplotlib.pyplot as plt


autodist = AutoDist(
    resource_spec_file='resource_spec.yml', 
    strategy_builder=PS(local_proxy_variable=False, sync=True, staleness=3)
)

d = autodist

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[:, :, :, None]
test_images = test_images[:, :, :, None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

BATCH_SIZE = 128

EPOCHS = 30
train_steps_per_epoch = min(1000, len(train_images) // BATCH_SIZE)
print(train_steps_per_epoch)

tf.reset_default_graph()
with tf.Graph().as_default(), d.scope():
    x = tf.keras.Input(shape=(28, 28, 1))
    y = tf.keras.Input(shape=())

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='auto')
    optimizer = tf.keras.optimizers.Adam()

    def train_step(x, y):
        y_hat = model(x, training=True)
        loss = loss_fn(y, y_hat)
        all_vars = []
        for v in model.trainable_variables:
            all_vars.append(v)
        grads = tf.gradients(loss, all_vars)
        update = optimizer.apply_gradients(zip(grads, all_vars))

        return loss, optimizer.iterations, update

    fetches = train_step(x, y)
    e_losses = []
    #init = tf.global_variables_initializer()
    sess = autodist.create_distributed_session()
    #sess = tf.Session()
    #sess.run(init)
    for epoch in range(EPOCHS):
        j = 0
        losses = []
        for _ in range(train_steps_per_epoch):
            loss, i, _ = sess.run(fetches, {x: train_images[j:j+BATCH_SIZE], y: train_labels[j:j+BATCH_SIZE]})
            #print(f"step: {i}, train_loss: {loss}")
            j += BATCH_SIZE
            losses.append(loss)
        e_losses.append(np.mean(losses))
        print(f"epoch: {epoch}, train_loss: {e_losses[epoch]}")


    plt.plot(range(EPOCHS), e_losses, 'bo', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./mygraph.png")

    sess.close()    

