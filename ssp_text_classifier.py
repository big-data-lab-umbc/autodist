import numpy as np
import matplotlib.pyplot as plt
from autodist.strategy import PS
import os
import sys

# IMDB - 2 layer network - train + validation + test, 20 epochs, 512 batch size (Tensorflow 1.x version)
# notes: 
# - uses keras to fetch the dataset for consistancy with other code samples, data operations and feeding to network done with numpy

import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorflow.python import debug as tf_debug

from autodist import AutoDist
autodist = AutoDist(
    resource_spec_file='resource_spec.yml', 
    strategy_builder=PS(local_proxy_variable=False, sync=True, staleness=3)
)

#network parameters
n_input = 10000 #input size for a single sample (10000 words)

#hyperparamters
batch_size = 512
eta = 0.001 # learning rate
max_epoch = 50

# 1. get data (using same dataset as keras example)
from keras.datasets import imdb

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=n_input)
np.load = np_load_old

#pre-process data into tensors
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test =  vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

y_train = y_train.reshape(len(y_train),1) #reshape for format taken by tf
y_test = y_test.reshape(len(y_test),1) #reshape for format taken by tf

#validation set to use during training
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#tf.reset_default_graph()
with tf.Graph().as_default(), autodist.scope():

    # 2. network architecture
    def multilayer_perceptron(x):
        #tf.reset_default_graph()
        fc1 = layers.fully_connected(x, 16, activation_fn = tf.nn.relu, scope = 'fc1')
        fc2 = layers.fully_connected(fc1, 16, activation_fn = tf.nn.relu, scope = 'fc2')
        out = layers.fully_connected(fc2, 1, activation_fn = None, scope = 'out') #no tf.Sigmoid activation function as tf.loss implementation expect raw output
        return out        

    # 3. select optimizer and loss

    #input data placeholders
    x = tf.placeholder(tf.float32, [None, n_input], name='placeholder_x')
    y = tf.placeholder(tf.float32, name='placeholder_y') 

    #network model
    y_hat = multilayer_perceptron(x)
    #loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y))
    #optimizer
    train = tf.train.RMSPropOptimizer(learning_rate= eta).minimize(loss)

    init = tf.global_variables_initializer()

    # 4. train / run network
    with autodist.create_distributed_session() as sess:
    #with tf.Session() as sess:
        sess.run(init)
        
        #define accuracy
        prediction = tf.nn.sigmoid(y_hat)
        correct_prediction = tf.equal(tf.round(prediction), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        losses = []
        acc = []
        val_acc = []
        
        for epoch in range(max_epoch):
            epoch_loss = 0.0
            batch_steps = int(len(partial_x_train) / batch_size)
            for i in range(batch_steps):
                batch_x = partial_x_train[i*batch_size:(i+1)*batch_size]
                batch_y = partial_y_train[i*batch_size:(i+1)*batch_size]
                _, c = sess.run([train, loss], feed_dict = {x: batch_x, y: batch_y })
                epoch_loss += c / batch_steps
            #validation_accuracy = accuracy.eval( {x: x_val, y: y_val}  )
            #train_accuracy = accuracy.eval( {x: partial_x_train, y: partial_y_train}  )

            losses.append(epoch_loss)
            #acc.append(train_accuracy)
            #val_acc.append(validation_accuracy)

            #print('Epoch %02d, Loss = %.6f, acc: %.6f, validation_acc: %.6f' % (epoch+1, epoch_loss, train_accuracy, validation_accuracy))
            print('Epoch %02d, Loss = %.6f' % (epoch+1, epoch_loss))
            
        # 5. test model
        #print("test_acc:", accuracy.eval( {x: x_test, y: y_test}  ))


        # "bo" is for "blue dot"
        plt.plot(range(max_epoch), losses, 'bo', label='Training loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # plt.plot(epoch, acc, 'bo', label='Training acc')
        # plt.plot(epoch, val_acc, 'b', label='Validation acc')
        # plt.title('Training and validation accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()

        plt.show()
        sess.close()