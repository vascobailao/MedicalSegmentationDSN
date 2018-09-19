import tensorflow as tf

class convolution_layer_3d():

    def __init__(self, layer_input, filter, strides, w, initializer):
        self.layer_input = layer_input
        self.filter = filter
        self.strides = strides
        self.padding = 'VALID'
        self.w = w
        self.initializer = initializer


    def convolution_layer_3d(self):
        assert len(self.filter) == 5  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
        assert len(self.strides) == 5  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
        assert self.padding in ['VALID', 'SAME']
        # w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')

        self.w = tf.Variable(initial_value=self.initializer(shape=filter), name='weights')

        b = tf.Variable(tf.constant(1.0, shape=[self.filter[-1]]), name='biases')
        convolution = tf.nn.conv3d(self.layer_input, self.w, self.strides, self.padding)
        return convolution + b