# coding: utf-8

import math
import numpy as np
import os
import pathlib
import random
import sys
import tensorflow as tf

import models
import datasets

# ---------------------------------------------------------------------------- #

# Main parameters
n          = 18
f          = 5
batch_size = 150

# Print main parameters
print("*** Running with n = %d, f = %d and b = %d ***" % (n, f, batch_size))

# ---------------------------------------------------------------------------- #

# Detailed parameters
m               = 200
assert m >= n
test_batch_size = 10000
do_cifar10      = True
learning_rate   = 0.002 if do_cifar10 else 0.05
activation_func = tf.nn.relu
max_train_step  = 100
max_train_accur = 0.97
load_parameters = False
save_parameters = False
parameters_path = pathlib.Path("model-" + ("cifar10" if do_cifar10 else "mnist") + ".npy")

# ---------------------------------------------------------------------------- #

# Constants
train_total_size = 50000 // batch_size * batch_size
byzcond_brute = (n - f) / (2 * f)
byzcond_krum  = 1 / math.sqrt(2 * (n - f + (f * (n - f - 2) + f * f * (n - f - 1)) / (n - 2 * f - 2)))
byzcond = [byzcond_brute, byzcond_krum]
byzcondsat = [0, 0]

def zeroes(grad, prop):
  """ Randomly zeros coordinates.
  Args:
    grad Gradient to mangle
    prop Proportion of the coordinates to zero
  Returns:
    Forwarded, mangled 'grad'
  """
  for i in range(len(grad)):
    if random.random() < prop:
      grad[i] = 0.
  return grad

# ---------------------------------------------------------------------------- #

# Dataset instantiation
if do_cifar10:
  dataset = datasets.load_cifar10()
else:
  dataset = datasets.load_mnist()
train_set = dataset.cut(0, train_total_size, train_total_size).shuffle().cut(0, train_total_size, batch_size)
test_set  = dataset.cut(50000, 60000, 10000).shuffle().cut(0, 10000, test_batch_size)

# Model instantiation
graph = tf.Graph()
with graph.as_default():
  builder_opt = tf.train.AdamOptimizer(learning_rate)
  if do_cifar10:
    model = models.conv_cifar10(act_fn=activation_func, optimizer=builder_opt)
  else:
    builder_dims = [784, 100, 10]
    model = models.dense_classifier(builder_dims, act_fn=activation_func, optimizer=builder_opt)

# Training
with graph.as_default():
  sess = tf.Session(graph=graph)
  with sess.as_default():
    sess.run(tf.global_variables_initializer())
    model.init()

    # Loading
    if load_parameters and parameters_path.exists():
      sys.stdout.write("Load model...")
      sys.stdout.flush()
      try:
        model.write(np.load(parameters_path))
        print(" done.")
      except Exception as err:
        print(" fail.")
        raise
    else:
      print("New model... done.")
    step = 0

    # Testing + training
    try:
      l = len(train_set)
      while True:
        # Testing
        sys.stdout.write("\rStep %3d: accuracy = " % step)
        sys.stdout.flush()
        acc = model.eval(*test_set.get())[0]
        print("%7.3f" % acc)
        # Check finished
        if acc >= max_train_accur:
          break
        if step >= max_train_step:
          break
        # Training
        sys.stdout.write("\rStep %3d:" % step)
        sys.stdout.flush()
        grads = [model.backprop(*train_set.get()) for _ in range(m)]
        model.update(sum(grads[:n]) / n)
        # Training
        gavg = sum(grads) / m
        gstd = math.sqrt(sum(np.linalg.norm(grad - gavg)**2 for grad in grads) / (m - 1))
        favg = float(np.linalg.norm(gavg))
        fstd = float(np.linalg.norm(gstd))
        fres = fstd / favg
        print(("%7.3f / %7.3f = %7.3f" % (fstd, favg, fres)) + " < (" + (", ").join(("%5.2f" % cond) + (" ☑" if fres < cond else " ☐") for cond in byzcond) + ")")
        for i in range(len(byzcond)):
          if fres < byzcond[i]:
            byzcondsat[i] += 1
        step += 1
    except KeyboardInterrupt:
      print("\rInterrupted at step " + str(step))

    # Print for how many steps condition was satisfied
    if step > 0:
      print("Progress condition satisfied: " + (", ").join(("%5.1f%%" % (satcnt / step * 100)) for satcnt in byzcondsat))

    # Saving (if some training done)
    if save_parameters and step > 0:
      sys.stdout.write("Saving...")
      sys.stdout.flush()
      try:
        np.save(parameters_path, model.read())
        print(" done.")
      except Exception as err:
        print(" fail (" + type(err).__name__ + ")")
