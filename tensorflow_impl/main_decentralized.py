import argparse

from Network import Network
from node_type.Worker import Worker
from node_type.PS import PS
from node_type.ByzWorker import ByzWorker

from aggregator_tf.aggregator import Aggregator_tf

import time
import os
import sys
import pickle

# Allowing visualization of the log while the process is running over ssh
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


FLAGS = None


def main():
    n_ps = Network(FLAGS.config_ps)
    n_w = Network(FLAGS.config_w)
        
    p = PS(n_ps, FLAGS.log, FLAGS.asyncr, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
    p.start()

    if n_w.get_my_attack() != 'None':
        w = ByzWorker(n_w, FLAGS.log, FLAGS.asyncr, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
    else:
        w = Worker(n_w, FLAGS.log, FLAGS.asyncr, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
    
    w.start()
        
    model_aggregator = Aggregator_tf('Median', len(n_w.get_all_workers()), FLAGS.nbbyzwrks)
    gradient_aggregator = Aggregator_tf(n_ps.get_my_strategy(), len(n_ps.get_all_workers()), FLAGS.nbbyzwrks)

    current_time = 0
    accuracy = 0
    accuracies = {}
    for iter in range(FLAGS.max_iter):
        start = time.time()
        models = w.get_models(iter)
        #aggregated_model = w.aggregate_models(models)
        aggregated_model = model_aggregator.aggregate(models)
        w.write_model(aggregated_model)
        p.write_model(aggregated_model)
        loss, grads = w.compute_gradients(iter)
        w.commit_gradients(grads)
            
        gradients = p.get_gradients(iter)
        #aggregated_gradient = p.aggregate_gradients(gradients)
        aggregated_gradient = gradient_aggregator.aggregate(gradients)
        model = p.upate_model(aggregated_gradient)
        p.commit_model(model)
        end = time.time()
        current_time += end - start
            
        if iter%50 == 0:
            print(iter)
        if iter%10 == 0:
            accuracy = p.compute_accuracy()
            accuracies[(iter, current_time)] = accuracy
    pickle.dump(accuracies, open('experiment_learn.p', 'wb'))
    p.wait_until_termination()
    w.wait_until_termination()
    #else:
    #    print("Unknown task type, please check TF_CONFIG file")
    #    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining current Node
    parser.add_argument('--config_w',
                        type=str,
                        default="TF_CONFIG",
                        help='Config file location.')
    parser.add_argument('--config_ps',
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
                        default="20",
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

    FLAGS, unparsed = parser.parse_known_args()
    main()
