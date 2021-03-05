from .server import Server

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam, SGD

from . import garfield_pb2
from . import tools

import time

class PS(Server):
    """ Parameter Server node, handles the updates of the parameter of the model. """

    def __init__(self, network=None, log=False, asyncr=False, dataset="mnist", model="Small", batch_size=128, nb_byz_worker=0, native=False):
        """ Create a Parameter Server node.

            args:
                - network:   State of the cluster
                - log:      Boolean indicating whether to log or not
                - asyncr:   Boolean

        """
        super().__init__(network, log, asyncr, dataset, model, batch_size, nb_byz_worker, native)
        
        lr=1e-3

        self.optimizer = Adam(lr=lr)



    def get_gradients(self, iter):
        """ Get gradients from the workers at a specific iteration.
        
            args:
                - iter: integer
            Returns:
                Gradients from the different PS.   
        """

        gradients = []

        for i, connection in enumerate(self.worker_connections):
            counter = 0
            read = False
            while not read:
                try:
                    response = connection.GetGradient(garfield_pb2.Request(iter=iter,
                                                                        job="ps",
                                                                        req_id=self.task_id))
                    serialized_gradient = response.gradients
                    gradient = np.frombuffer(serialized_gradient, dtype=np.float32)
                    gradients.append(gradient)
                    read = True
                except Exception as e:
                    print("Trying to connect to Worker node ", i)
                    time.sleep(5)
                    counter+=1
                    if counter > 10:			#any reasonable large enough number
                        exit(0)
        return gradients

    def upate_model(self, gradient):#, gradients, lr, epoch):
        """ Update the model with the aggregated gradients. 

            Args:
                - gradient: gradient to update the model.
            Returns:
                Model after update.        
        """
        reshape_gradient = tools.reshape_weights(self.model, gradient)
        self.optimizer.apply_gradients(zip(reshape_gradient, self.model.trainable_variables))
        return tools.flatten_weights(self.model.trainable_variables)

    def commit_model(self, model):
        """ Make the model available on the network. 

            Args:
                - model: model to commit
        """
        self.service.model_wieghts_history.append(model)
