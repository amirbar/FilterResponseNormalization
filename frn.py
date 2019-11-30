from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K


def FRNLayer(x, tau, beta, gamma, eps=1e-6):
    # x: Input tensor of shape [BxHxWxC].
    # alpha, beta, gamma: Variables of shape [1, 1, 1, C].
    # eps: A scalar constant or learnable variable.
    # Compute the mean norm of activations per channel.
    nu2 = tf.reduce_mean(tf.square(x), axis=range(1, x.ndim-1), keepdims=True)
    # Perform FRN.
    x = x * tf.rsqrt(nu2 + tf.abs(eps))
    # Return after applying the Offset-ReLU non-linearity.
    return tf.maximum(gamma * x + beta, tau)


class FRN(Layer):

    def __init__(self,
                 axis=-1,
                 epsilon=1e-6,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 tau_initializer='zeros',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 tau_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 tau_constraint=None,
                 **kwargs):
        super(FRN, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.tau_initializer = initializers.get(tau_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.tau_regularizer = regularizers.get(tau_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.tau_constraint = constraints.get(tau_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape,
                                     name='gamma',
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)

        self.tau = self.add_weight(shape=shape,
                                    name='tau',
                                    initializer=self.tau_initializer,
                                    regularizer=self.tau_regularizer,
                                    constraint=self.tau_constraint)

        self.built = True

    def call(self, inputs, **kwargs):
        return FRNLayer(inputs, gamma=self.gamma, beta=self.beta, tau=self.tau)

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'tau_initializer': initializers.serialize(self.tau_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'tau_regularizer': regularizers.serialize(self.tau_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'tau_constraint': constraints.serialize(self.tau_constraint)
        }
        base_config = super(FRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
