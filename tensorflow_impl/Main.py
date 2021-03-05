import argparse

from Network import Network
from node_type.Worker import Worker
from node_type.PS import PS
from node_type.ByzWorker import ByzWorker

from aggregator_tf.aggregator import Aggregator_tf

import pickle
import time
import os
import sys
from node_type import tools
# Allowing visualization of the log while the process is running over ssh
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)



FLAGS = None


def main():
    n = Network(FLAGS.config)

    if n.get_task_type() == 'worker':
        if n.get_my_attack() != 'None':
            w = ByzWorker(n, FLAGS.log, FLAGS.asyncr, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks, FLAGS.native)
        else:
            w = Worker(n, FLAGS.log, FLAGS.asyncr, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks, FLAGS.native)
        w.start()
        model_aggregator = Aggregator_tf('Median', len(n.get_all_workers()), FLAGS.nbbyzwrks)

        for iter in range(FLAGS.max_iter):
            models = w.get_models(iter)
            aggregated_model = model_aggregator.aggregate(models)
            w.write_model(aggregated_model)
            loss, grads = w.compute_gradients(iter)
            w.commit_gradients(grads)

        w.wait_until_termination()

    elif n.get_task_type() == 'ps':
        p = PS(n, FLAGS.log, FLAGS.asyncr, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks, FLAGS.native)
        p.start()

        model_aggregator = Aggregator_tf('Median', len(n.get_all_workers()), FLAGS.nbbyzwrks)
        gradient_aggregator = Aggregator_tf(n.get_my_strategy(), len(n.get_all_workers()), FLAGS.nbbyzwrks)

        current_time = 0
        accuracy = 0
        accuracies = {}
        for iter in range(FLAGS.max_iter):
            start = time.time()
            models = p.get_models(iter)
            aggregate_model = model_aggregator.aggregate(models)
            p.write_model(aggregate_model)
            gradients = p.get_gradients(iter)
            aggregated_gradient = gradient_aggregator.aggregate(gradients)
            model = p.upate_model(aggregated_gradient)
            p.commit_model(model)
            end = time.time()
            tools.training_progression(FLAGS.max_iter, iter, accuracy)
            if iter%50 == 0:
                accuracy = p.compute_accuracy()
                accuracies[(iter, current_time)] = accuracy

            current_time += end - start

        pickle.dump(accuracies, open('experiment.p', 'wb'))
        print("\nTraining done!")
        p.wait_until_termination()
    else:
        print("Unknown task type, please check TF_CONFIG file")
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Flags for defining current Node
    parser.add_argument('--config',
                        type=str,
                        default="TF_CONFIG",
                        help='Config file location.')
    parser.add_argument('--log',
                        type=bool,
                        default=False,
                        help='Add flag to print intermediary steps.')
    parser.add_argument('--asyncr',
                        type=bool,
                        default=False,
                        help='Add flag to indicate that the network is asynchronous.')
    parser.add_argument('--max_iter',
                        type=int,
                        default="2000",
                        help='Maximum number of epoch')
    parser.add_argument('--dataset',
                        type=str,
                        default="mnist",
                        help='Choose the dataset to use')
    parser.add_argument('--model',
                        type=str,
                        default="Small",
                        help='Choose the model to use')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Set the batch size')
    parser.add_argument('--nbbyzwrks',
                        type=int,
                        default=0,
                        help='Set the number of byzantine workers (necessary for Krum aggregation)')
    parser.add_argument('--native',
                        type=bool,
                        default=False,
                        help='Choose to use the native aggregators.')


    FLAGS, unparsed = parser.parse_known_args()
    main()
