import tensorflow.compat.v1 as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.datasets import imdb

import numpy as np
from autodist import AutoDist
from autodist.strategy import PS


autodist = AutoDist(
    resource_spec_file='resource_spec.yml', 
    strategy_builder=PS(local_proxy_variable=False, sync=True, staleness=1)
    #strategy_builder=PS()
)

d = autodist

#network parameters
n_input = 800 #input size for a single sample (800 words)

#hyperparamters
batch_size = 128
eta = 0.001 # learning rate
max_epoch = 20

# 1. get data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=800)

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

#validation set to use during training
x_val = x_train[:800]
partial_x_train = x_train[800:]

y_val = y_train[:800]
partial_y_train = y_train[800:]

# 2. network architecture
class IMDBModel(tf.keras.Model):
    def __init__(self):
        #constructor = define all layers (without connecting them)
        super(IMDBModel, self).__init__()
        self.fc1 = Dense(16, activation='relu')
        self.fc2 = Dense(16, activation='relu')
        self.out = Dense(1, activation='sigmoid')

    def call(self, x):
        #connect layers / tell the model the order of execution of layers
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

tf.reset_default_graph()
with tf.Graph().as_default(), d.scope():    

    model = IMDBModel()

    # 3. select optimizer and loss
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.RMSprop()

    #define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    # 4. train / run network

    #define training procedure
    @d.function #this makes python code compile to tensorflow C backend
    def train_step(batch_x, batch_y):
        #run forward pass
        with tf.GradientTape() as tape: #gradient tape is used to "record" forward pass operations
            predictions = model(batch_x) #(1) execute all layers (reference the "call" method of the subclassed IMDBModel)
            loss = loss_object(y_true = batch_y, y_pred = predictions) #(2) calculate loss comparing labels vs. model output

        #run backpropagation, calculate and apply gradients to adjust model weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #calcualte metrics
        train_loss(loss)
        train_accuracy(batch_y, predictions)

    #define test procedure
    @d.function
    def test_step(batch_x, batch_y):
        predictions = model(batch_x)
        t_loss = loss_object(y_true = batch_y, y_pred = predictions)

        test_loss(t_loss)
        test_accuracy(batch_y, predictions)

    session = autodist.create_distributed_session()

    #run train    
    for epoch in range(max_epoch):
        batch_steps = int(len(partial_x_train) / batch_size)
        for i in range(batch_steps):
            batch_x = partial_x_train[i*batch_size:(i+1)*batch_size]
            batch_y = partial_y_train[i*batch_size:(i+1)*batch_size]
            fetches_train = train_step(batch_x, batch_y)
            session.run(fetches_train)

        #check validation accuracy
        fetches_test = test_step(x_val, y_val)
        session.run(fetches_test)

        template = 'Epoch {}, loss: {} - acc: {} - val_loss: {} - val_acc: {}'
        print(template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result(),
                            test_loss.result(),
                            test_accuracy.result()))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        
    # # 5. test model

    # test_loss.reset_states()
    # test_accuracy.reset_states()
    # for i in range(batch_steps):
    #     batch_x = x_test[i*batch_size:(i+1)*batch_size] #, tf.newaxis
    #     batch_y = y_test[i*batch_size:(i+1)*batch_size] #, tf.newaxis
    #     test_step(batch_x, batch_y)
        
    # print("test_acc: {}".format(test_accuracy.result()))

    session.close()