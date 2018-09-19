import tensorflow as tf
from MedicalSegmentationDSN.src.network.initializers.Initializer import xavier_uniform_dist_conv3d

def deconvolution_layer_3d(layer_input, filter, output_shape, strides, initializer=xavier_uniform_dist_conv3d, padding='SAME'):
    assert len(filter) == 5  # [depth, height, width, output_channels, in_channels]
    assert len(strides) == 5  # must match input dimensions [batch, depth, height, width, in_channels]
    assert padding in ['VALID', 'SAME']
    # w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')
    w = tf.Variable(initial_value=initializer(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')
    deconvolution = tf.nn.conv3d_transpose(layer_input, w, output_shape, strides, padding)
    return deconvolution + b