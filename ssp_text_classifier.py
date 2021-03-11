import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import os

from tensorflow.python import debug as tf_debug
from autodist import AutoDist
from autodist.strategy import PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax

resource_spec_file = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')
autodist = AutoDist(resource_spec_file, PS())

tf.reset_default_graph()

#network parameters
n_input = 800 #input size for a single sample (800 words)

#hyperparamters
batch_size = 512
eta = 0.001 # learning rate
max_epoch = 10

# 1. get data (using same dataset as keras example)
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=n_input)

#pre-process data into tensors
def vectorize_sequences(sequences, dimension=800):
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
x_val = x_train[:800]
partial_x_train = x_train[800:]

y_val = y_train[:800]
partial_y_train = y_train[800:]

e_test_accuracy = []


with tf.Graph().as_default(), autodist.scope():
    # 2. network architecture
    def multilayer_perceptron(x):
        #tf.reset_default_graph()
        fc1 = layers.fully_connected(x, 16, activation_fn = tf.nn.relu, scope = 'fc1')
        fc2 = layers.fully_connected(fc1, 16, activation_fn = tf.nn.relu, scope = 'fc2')
        out = layers.fully_connected(fc2, 1, activation_fn = None, scope = 'out') #no tf.Sigmoid activation function as tf.loss implementation expect raw output
        return out

    x = tf.placeholder(tf.float32, [None, n_input], name='placeholder_x')
    y = tf.placeholder(tf.float32, name='placeholder_y') 

    def train_step(x,y):
        y_hat = multilayer_perceptron(x)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y))
        train = tf.train.RMSPropOptimizer(learning_rate= eta).minimize(loss)
        return train, loss

    y_hat = multilayer_perceptron(x)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y))
    train = tf.train.RMSPropOptimizer(learning_rate= eta).minimize(loss)

    init = tf.global_variables_initializer()

    prediction = tf.nn.sigmoid(y_hat)
    correct_prediction = tf.equal(tf.round(prediction), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with autodist.create_distributed_session() as sess:
        sess.run(init)
                
        for epoch in range(max_epoch):
            epoch_loss = 0.0
            batch_steps = int(len(partial_x_train) / batch_size)
            for i in range(batch_steps):
                batch_x = partial_x_train[i*batch_size:(i+1)*batch_size]
                batch_y = partial_y_train[i*batch_size:(i+1)*batch_size]
                _, c = sess.run([train, loss], feed_dict = {x: batch_x, y: batch_y })
                epoch_loss += c / batch_steps
                validation_accuracy = sess.run([accuracy], feed_dict = {x: x_val, y: y_val})
                e_test_accuracy.append(validation_accuracy[0])

            print('Epoch %02d, Loss = %.6f, validation_acc: %.6f' % (epoch+1, epoch_loss, validation_accuracy[0]))

        print("\nTest_accuracy,",", ".join([str(x) for x in e_test_accuracy]))
        sess.close()
