import tensorflow as tf
from MedicalSegmentationDSN.src.network.initializers.Initializer import xavier_normal_dist_conv3d


def convolution_layer_3d(layer_input, filter, strides, initializer=xavier_normal_dist_conv3d, padding='SAME'):
    assert len(filter) == 5  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 5  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']
    # w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')

    w = tf.Variable(initial_value=initializer(shape=filter), name='weights')

    b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
    convolution = tf.nn.conv3d(layer_input, w, strides, padding)
    return convolution + b