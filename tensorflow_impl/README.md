# Garfield_TF
Building Garfield library on TensorFlow

## Structure

```
├───aggregator          (different aggregation methods)
└───node_type           (definitions for worker/PS process)
```

## Usage

In order to run a process, you need to have a `TF_CONFIG` file for _each_ process in the cluster.

### Setting TF_CONFIG file

This file inform a process of the different nodes in the cluster, as well as its task.

##### Cluster declaration:

This part needs to be the same for all config file in the cluster.

```
"cluster": {
    "worker": ["host0:port0", "host1:port1", "host2:port2"],
    "ps": ["host3:port3", "host4:port4"]
}
```

##### Task declaration:

This part must be unique for each process. You need to define if your process is a `ps` or a `worker`. The index informs the position of the process's ip in the IP:PORT list (starting at 0).

```
"task": {
  "type": "ps",
  "index": 0,
  "strategy": "Mean",  (Aggregation strategy, choose from: Mean, Median)
  "attack": "None"     (Define if the process is bizantine or not, choose from: None, Reverse)
}
```

### Starting a process

Run `Main.py` to start a process

```
usage: runner.py [-h] --cluster CLUSTER [--log] [--epoch EPOCH]
                 [--batch_size BATCH_SIZE]
                 [--dataset DATASET]
                 
arguments:
  -h, --help                show this help message and exit
  --cluster CLUSTER         path of the TF_CONFIG file
  --log                     display intermediary steps
  --epoch EPOCH             number of epoch
  --dataset DATASET         choose the dataset to use
  --batch_size BATCH_SIZE   chosse the batch size to use

```

## Requirements

All requirements are listed in the requirement file
