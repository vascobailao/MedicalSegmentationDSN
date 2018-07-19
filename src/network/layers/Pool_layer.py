import numpy as np
import tensorflow as tf

class Pool_layer():

    def __init__(self, layer_input, ksize, strides, padding):
        self.layer_input = layer_input
        self.ksize = ksize
        self.strides = strides
        self.padding = 'VALID'


    def max_pooling_3d(self):
        assert len(self.ksize) == 5  # [batch, depth, rows, cols, channels]
        assert len(self.strides) == 5  # [batch, depth, rows, cols, channels]
        assert self.ksize[0] == self.ksize[4]
        assert self.ksize[0] == 1
        assert self.strides[0] == self.strides[4]
        assert self.strides[0] == 1
        return tf.nn.max_pool3d(self.layer_input, self.ksize, self.strides, self.padding)