import sys
import numpy as np
import os
import tensorflow.compat.v1 as tf
import keras

from autodist import AutoDist
from autodist.strategy import PS
#import matplotlib.pyplot as plt


autodist = AutoDist(
    resource_spec_file='resource_spec.yml', 
    strategy_builder=PS(local_proxy_variable=False, sync=True, staleness=1)
    #strategy_builder=PS()
)

d = autodist

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[:8000, :, :, None]     
#train_images = train_images[8000:16000, :, :, None]
#train_images = train_images[16000:24000, :, :, None]
#train_images = train_images[24000:32000, :, :, None]
#train_images = train_images[32000:40000, :, :, None]
#train_images = train_images[40000:48000, :, :, None]
#train_images = train_images[48000:56000, :, :, None]

test_images = test_images[:4000, :, :, None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

BATCH_SIZE = 512

EPOCHS = 20
train_steps_per_epoch = min(1000, len(train_images) // BATCH_SIZE) #len(train_images)=60000

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

        return loss, optimizer.iterations, update, y, y_hat

    def test_step(x, y):
        y_hat = model(x, training=True)
        
        return y, y_hat

    fetches = train_step(x, y)
    test_fetches = test_step(x, y)
    e_losses = []
    e_train_accuracy = []
    e_test_accuracy = []

    #sess = autodist.create_distributed_session()
    with autodist.create_distributed_session() as sess:
    #sess = tf.Session()
        for epoch in range(EPOCHS):
            j = 0
            losses = []
            for _ in range(train_steps_per_epoch):
                loss, i, _, prediction, prediction_hat = sess.run(fetches, {x: train_images[j:j+BATCH_SIZE], y: train_labels[j:j+BATCH_SIZE]})
                #print(f"step: {i}, train_loss: {loss}")
                j += BATCH_SIZE
                losses.append(loss)
            e_losses.append(np.mean(losses))
            
            prediction, prediction_hat = sess.run(test_fetches, {x: train_images[0:4000], y: train_labels[0:4000]})
            prediction = keras.utils.to_categorical(prediction, 10)
            correct_prediction = tf.equal(tf.round(prediction_hat), prediction)
            train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).eval()
            e_train_accuracy.append(train_accuracy)

            prediction, prediction_hat = sess.run(test_fetches, {x: test_images, y: test_labels})
            prediction = keras.utils.to_categorical(prediction, 10)
            correct_prediction = tf.equal(tf.round(prediction_hat), prediction)
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).eval()
            e_test_accuracy.append(test_accuracy)

            #print("\n\nepoch: ",epoch,"\nloss: ",e_losses[epoch],"\ntrain_accuracy: ",train_accuracy,"\ntest_accuracy: ",test_accuracy)

    # listToStr = ' '.join([str(elem) for elem in e_losses])

    # plt.plot(range(EPOCHS), e_losses, 'bo', label='Training loss')
    # #plt.title('Training loss: ' + listToStr)
    # plt.title('Training loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig("./average.png")

    print("train_loss = ",e_losses,"\ntrain_accuracy = ",e_train_accuracy,"\ntest_accuracy = ",e_test_accuracy)

    sess.close()    

