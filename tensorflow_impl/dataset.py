import tensorflow_datasets as tfds
import tensorflow as tf


class DatasetManager:

    def __init__(self, network, dataset, batch_size=128):
        available_dataset = ['mnist', 'cifar10', 'cifar100']
        if dataset not in available_dataset:
            raise Exception("Dataset unavailable, please select from available dataset: " + str(available_dataset))

        
        split_low, split_high = self.get_data_partition(network)[network.get_task_index()]
        print(split_low, split_high)
        (ds_train, ds_test), ds_info = tfds.load(
            dataset,
            split=['train[{}%:{}%]'.format(split_low, split_high), 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        self.data_train, self.data_test = self.process_data(ds_train, ds_test, ds_info, batch_size)
        self.input_size = (28, 28, 1) if dataset == 'mnist' else (32, 32, 3)
        self.classes = 10 if dataset != 'cifar100' else 100

    def normalize_img(self, image, label):
        return tf.cast(image, tf.float32) / 255.0, tf.cast(label, tf.float32)

    def process_data(self, ds_train, ds_test, ds_info, batch_size):
        ds_train = ds_train.map(
            self.normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        ds_train = list(tfds.as_numpy(ds_train))

        ds_test = ds_test.map(
            self.normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.batch(600)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        ds_test = list(tfds.as_numpy(ds_test))

        return ds_train, ds_test

    def get_data_partition(self, network):
        number_worker = len(network.get_all_workers())
        partition_size = round(100 / number_worker)

        partition = {}
        starting = 0

        for worker_task in range(number_worker):
            partition[worker_task] = (starting, min(100, starting + partition_size))
            starting = starting + partition_size

        return partition
