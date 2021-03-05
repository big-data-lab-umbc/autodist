from .server import Server

from . import tools

import tensorflow as tf
import numpy as np

from . import garfield_pb2

import time


class Worker(Server):
    """ Worker node used to calculate the gradient of a model. """

    def __init__(self, network=None, log=False, asyncr=False, dataset="mnist", model="Simple", batch_size=128,
                 nb_byz_worker=0, native=False):
        """ Create a Worker node.

            args:
                - network:  State of the cluster
                - log:      Boolean indicating whether to log or not
                - asyncr:   Boolean

        """
        super().__init__(network, log, asyncr, dataset, model, batch_size, nb_byz_worker, native)

    def compute_gradients_for_CPU(self, iter):
        """ Compute gradients. 
        
            Args:
                - iter: iteration of the training
            Returns:
                Gradient of the model based on the data of a specific iteration.
        """

        X, y = self.data[iter % len(self.data)]

        with tf.GradientTape() as tape:
            preds = self.model(X, training=True)

            loss = self.loss_fn(y, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)

        return loss, tools.flatten_weights(grads)

    def compute_gradients(self, iter):
        """ Compute gradients.

            Args:
                - iter: iteration of the training
            Returns:
                Gradient of the model based on the data of a specific iteration.
        """

        if tf.config.list_physical_devices('gpu'):
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()

        X, y = self.data[iter % len(self.data)]
        dataset = tf.data.Dataset.from_tensors((X, y))
        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        with strategy.scope():
            @tf.function
            def train_step(inputs):
                features, labels = inputs

                with tf.GradientTape() as tape:
                    preds = self.model(features, training=True)

                    loss = tf.reduce_sum(self.loss_fn(labels, preds)) * (1. / self.batch_size)


                grads = tape.gradient(loss, self.model.trainable_variables)
                flattened = tools.flatten_weights(grads)

                return loss, flattened

            @tf.function
            def distributed_train_step(dist_inputs):
                per_replica_losses, per_replica_grad = strategy.run(train_step, args=(dist_inputs,))
                return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), \
                       strategy.experimental_local_results(value=per_replica_grad)

            tf.config.run_functions_eagerly(True)
            losses = []
            grads = []
            for dist_inputs in dist_dataset:
                l, g = distributed_train_step(dist_inputs)
                losses.append(l)
                grads.append(g)

            return np.mean(losses), np.mean(grads, axis=0)

    def commit_gradients(self, grads):
        """ Make the gradients available to the other nodes on the network.
        
            Args:
                - grads: Computed gradient.
        """

        self.service.gradients_history.append(grads)
