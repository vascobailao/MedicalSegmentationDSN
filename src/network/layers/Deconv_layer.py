import tensorflow as tf

class Deconv_layer():

    def __init__(self, layer_input, filter, output_shape, strides, w, b, initializer):
        self.layer_input = layer_input
        self.filter = filter
        self.output_shape = output_shape
        self.strides = strides
        self.padding = 'SAME'
        self.weights = w
        self.biases = b
        self.initializer = initializer


    def deconvolution_layer_3d(self):
        assert len(self.filter) == 5  # [depth, height, width, output_channels, in_channels]
        assert len(self.strides) == 5  # must match input dimensions [batch, depth, height, width, in_channels]
        assert self.padding in ['VALID', 'SAME']
        # w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')
        self.w = tf.Variable(initial_value=self.initializer(shape=self.filter), name='weights')
        self.b = tf.Variable(tf.constant(1.0, shape=[self.filter[-2]]), name='biases')
        deconvolution = tf.nn.conv3d_transpose(self.layer_input, self.w, self.output_shape, self.strides, self.padding)
        return deconvolution + self.b