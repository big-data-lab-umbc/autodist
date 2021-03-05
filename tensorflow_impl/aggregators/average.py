# coding: utf-8
###
 # @section DESCRIPTION
 #
 # Simple synchronous average GAR.
###

import tensorflow as tf

from . import _GAR, register

# ---------------------------------------------------------------------------- #
# Average GAR class

class AverageGAR(_GAR):
  """ Simple synchronous average GAR class.
  """

  def __init__(self, nbworkers, nbbyzwrks, args):
    pass

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    if len(gradients) > 1:
      return tf.add_n(gradients) / float(len(gradients))
    else:
      return gradients[0]

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
register("average", AverageGAR)
