import numpy as np
import tensorflow as tf

a = tf.constant([[3., 5.], [4., 8.]])
b = tf.constant([[1., 6.], [2., 9.]])

# gradients = [a,b]

# # n_gradients = [a.eval(session=tf.compat.v1.Session()), b.eval(session=tf.compat.v1.Session())]   
# # print(n_gradients)
# #print(np.median(n_gradients, axis=0))  #[[2.  5.5],[3.  8.5]]

# # print(gradients)

# with tf.Session() as sess:
#     result = sess.run(tf.math.reduce_mean(gradients,0))
#     print(result)
# print(tf.math.reduce_mean(gradients,0))

print(a.shape)