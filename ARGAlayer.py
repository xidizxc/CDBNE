# -*- coding: utf-8 -*-
# @Author  : ZXC
# @Time    : 2021/11/9 10:54
# @Function:
import torch as tr
from gae.initializations import weight_variable_glorot
from gae.layers import dropout_sparse

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    def __init__(self,**kwargs):
        allowed_kwargs = {'name','logging'}
        for kwarg in kwargs.keys():
            assert kwargs in allowed_kwargs, 'Invalid keyword argument: '+kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__lower()
            name = layer+'_'+str(get_layer_uid(layer))
            self.name=name
            self.vars={}
            logging = kwargs.get('logging',False)
            self.logging=logging
            self.issparse=False
    def _call(self,inputs):
        return inputs
    def __call__(self, inputs):
        with tr.name_scope(self.name):
            outputs=self._call(inputs)
            return  outputs

class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tr.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tr.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tr.nn.dropout(x, 1-self.dropout)
        x = tr.matmul(x, self.vars['weights'])
        x = tr.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tr.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tr.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tr.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tr.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tr.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tr.nn.dropout(inputs, 1-self.dropout)
        x = tr.transpose(inputs)
        x = tr.matmul(inputs, x)
        x = tr.reshape(x, [-1])
        outputs = self.act(x)
        return outputs