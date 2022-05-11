# -*- coding: utf-8 -*-
# @Author  : ZXC
# @Time    : 2021/11/9 10:53
# @Function:
import torch as tr
from layer import GATLayer

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tr.variable_scope(self.name):
            self._build()
        variables = tr.get_collection(tr.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class ARGA(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(ARGA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        with tr.variable_scope('Encoder', reuse=None):
            self.hidden1 = GraphConvolution(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(self.inputs)

            self.noise = gaussian_noise_layer(self.hidden1, 0.1)

            self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                               output_dim=FLAGS.hidden2,
                                               adj=self.adj,
                                               act=lambda x: x,
                                               dropout=self.dropout,
                                               logging=self.logging,
                                               name='e_dense_2')(self.noise)

            self.z_mean = self.embeddings

            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                       act=lambda x: x,
                                                       logging=self.logging)(self.embeddings)

def gaussian_noise_layer(input_layer, std):
    noise = tr.random_normal(shape=tr.shape(input_layer), mean=0.0, stddev=std, dtype=tr.float32)
    return input_layer + noise