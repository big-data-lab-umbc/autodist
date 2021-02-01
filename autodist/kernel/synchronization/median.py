# coding: utf-8
###
 # @file   median.py
 # @author Sébastien Rouault <sebastien.rouault@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 Sébastien ROUAULT.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in all
 # copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 # @section DESCRIPTION
 #
 # Coordinate-wise median GAR.
###

import tensorflow as tf

from . import deprecated_native

# ---------------------------------------------------------------------------- #
# Median coordinate-per-coordinate GAR class

def _aggregate(gradients):
  """ Aggregate the gradient using the associated deprecated_native helper.
  Args:
    gradients Stacked list of submitted gradients, as a numpy array
  Returns:
    Aggregated gradient, as a numpy array
  """
  gradients = tf.parallel_stack(gradients)
  g = [gradients]
  return deprecated_native.median(g)

def aggregate(gradients, tmp):
  # Assertion
  assert len(gradients) > 0, "Empty list of gradient to aggregate"
  # Computation
  #gradients = tf.parallel_stack(gradients)
  #tmp = tf.py_func(type(self)._aggregate, gradients, gradients[0].dtype, stateful=False, name="GAR_median")
  return tmp

