import node_type.tools as tools
from aggregator_tf.aggregator import Aggregator_tf
# import tensorflow_datasets as tfds


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization

import time

from concurrent import futures

import grpc

from dataset import DatasetManager
from model import ModelManager
from . import garfield_pb2_grpc
from . import garfield_pb2
from . import grpc_message_exchange_servicer


class Server:
    """ Superclass defining a server entity. """

    def __init__(self, network=None, log=False, asyncr=False, dataset="mnist", model="Small", batch_size=128, nb_byz_worker=0, native=False):
        self.log = log
        self.asyncr = asyncr
        self.aggregator = Aggregator_tf(network.get_my_strategy(), len(network.get_all_workers()), nb_byz_worker, native)
        # self.aggregator = aggregators.instantiate(network.get_my_strategy(), len(network.get_all_workers()), nb_byz_worker, None)

        self.network = network
        self.nb_byz_worker = nb_byz_worker
        self.batch_size = batch_size

        dsm = DatasetManager(network, dataset, batch_size)
        self.data, self.test_data = dsm.data_train, dsm.data_test

        mdm = ModelManager(model=model, input_shape=dsm.input_size, classes=dsm.classes)

        self.model = mdm.model

        ps_hosts = network.get_all_ps()
        worker_hosts = network.get_all_other_worker()
        self.ps_connections = [tools.set_connection(host) for host in ps_hosts]
        self.worker_connections = [tools.set_connection(host) for host in worker_hosts]

        self.port = network.get_my_port()
        self.task_id = network.get_task_index()

        self.m = tf.keras.metrics.Accuracy()
        
        # Define grpc server
        self.service = grpc_message_exchange_servicer.MessageExchangeServicer(tools.flatten_weights(self.model.trainable_variables))

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=30), options=[
            ('grpc.max_send_message_length', 500 * 1024 * 1024),
            ('grpc.max_receive_message_length', 500 * 1024 * 1024)
        ])
        garfield_pb2_grpc.add_MessageExchangeServicer_to_server(self.service, self.server)
        self.server.add_insecure_port('[::]:' + str(self.port))

        self.aggregated_weights = None

        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)



    def start(self):
        """ Starts the gRPC server. """

        self.server.start()
        if self.log:
            print("Starting on port: " + str(self.port))

    def wait_until_termination(self):
        self.server.wait_for_termination()

    def get_models(self, iter):
        """ Get all the models of the parameter servers. 
        
            args:
                - iter: int the iteration of the training

            returns:
                A list of model
        """
        models = []
        
        for i, connection in enumerate(self.ps_connections):
            counter = 0
            read = False
            while not read:
                try:
                    response = connection.GetModel(garfield_pb2.Request(iter=iter,
                                                                job="worker",
                                                                req_id=self.task_id))
                    serialized_model = response.model
                    model = np.frombuffer(serialized_model, dtype=np.float32)
                    models.append(model)
                    read = True
                except Exception as e:
                    print("Trying to connect to PS node ", i)
                    time.sleep(5)
                    counter+=1
                    if counter > 100:			#any reasonable large enough number
                        exit(0)
            
        return models

    def write_model(self, model):
        """ Build a Keras model from flatten weights. """

        for l, weights in zip(self.model.trainable_variables, tools.reshape_weights(self.model,model)):
            l.assign(weights.reshape(l.shape))


    def compute_accuracy(self):
        """ Compute the accuracy of the model on the test set and print it. """
        predictions = []
        true_val = []
        for X, y in self.test_data:
            preds = self.model(X)
            predictions = predictions + [float(tf.argmax(p).numpy()) for p in preds]
            true_val.extend(y)

        self.m.reset_states()
        self.m.update_state(y_pred=predictions, y_true=true_val)
        #print("Accuracy K: {}%".format(m.result().numpy() * 100))
        return self.m.result().numpy() * 100
