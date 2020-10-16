from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import tensorflow_hub as hub


class ElmoEmbeddingLayer(Layer):
    def __init__(self, trainable, **kwargs):
        self.dimensions = 1024
        self.trainable = trainable
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
            self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module2".format(self.name))
            self.trainable_weights += tf.compat.v1.trainable_variables(scope="^{}_module2/.*".format(self.name))
            super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
            result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                          as_dict=True,
                          signature='default',
                          )['elmo']
            return result

    # def compute_mask(self, inputs, mask=None):
    #         return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
            return (input_shape[0], None, self.dimensions)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(ElmoEmbeddingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


