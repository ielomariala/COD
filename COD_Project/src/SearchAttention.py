# Version 1
from tensorflow.keras.layers import Attention

def SearchAttention_v1(attention, x):
    attention = Conv2D(x.shape[-1], kernel_size=1, use_bias=False)(attention)
    return Attention()([attention, x])

# Version 2
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, ZeroPadding2D

import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import scipy.stats as st

def _get_kernel(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def gaussian_kernel():
  gaussian_kernel = np.float32(_get_kernel(31, 4))
  gaussian_kernel = gaussian_kernel[... , np.newaxis, np.newaxis]
  return gaussian_kernel

def min_max_norm(x):

    shape = tf.shape(x)
    # print(x, shape)

    max_ = tf.math.reduce_max(x, axis=2)[0]
    # print("Passed max 2", max_, tf.shape(max_))
    max_ = tf.math.reduce_max(max_, axis=1)[0]
    # print("Passed max 1", max_, tf.shape(max_))
    max_ = tf.expand_dims(max_, -1)
    # print("Passed max expand dims 1", max_, tf.shape(max_))
    max_ = tf.expand_dims(max_, -2)
    # print("Passed max expand dims 2", max_, tf.shape(max_))
    max_ = tf.broadcast_to(max_, shape)
    # print("Passed max broadcast", max_, tf.shape(max_))

    min_ = tf.math.reduce_min(x, axis=2)[0]
    # print("Passed min 2", min_, tf.shape(min_))
    min_ = tf.math.reduce_min(min_, axis=1)[0]
    # print("Passed min 1", min_, tf.shape(min_))
    min_ = tf.expand_dims(min_, -1)
    # print("Passed min expand dims 1", min_, tf.shape(min_))
    min_ = tf.expand_dims(min_, -2)
    # print("Passed min expand dims 2", min_, tf.shape(min_))
    min_ = tf.broadcast_to(min_, shape)
    # print("Passed min broadcast", min_, tf.shape(min_))


    x = x - min_
    # print("Passed min subtraction", x, tf.shape(x))
    x = x / (max_ - min_ + 1e-8)
    # print("Passed division", x, tf.shape(x))
    return x
    

def SearchAttention_v2(attention, x):

    padding_list = [[0, 0], [15, 15],[15, 15], [0, 0]]
    # attention = tf.constant(attention, dtype=tf.float32)
    soft_attention = tf.nn.conv2d(attention, gaussian_kernel(), padding=padding_list, strides=1)
    # print("Passed convolution")
    
    soft_attention = min_max_norm(soft_attention)       # normalization
    # print("Passed normalization")
    m = tf.math.maximum(soft_attention, attention)
    # print("Passed max")
    ret = tf.math.multiply(x, m)                        # multiplication
    # print("Passed multiplication")
    return ret