import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import tensorflow.compat.v1 as tf
from gae.initializations import weight_variable_glorot



from layer import GATELayer
# flags = tf.app.flags
# FLAGS = flags.FLAGS

class GATE(nn.Module):
    def __init__(self, attribute_number, hidden_size, embedding_size, alpha):
        super(GATE, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATELayer(attribute_number, hidden_size, alpha)
        self.conv2 = GATELayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        # x: [samples_cnt=N, input_feat]
        # w: [input_feat, output_feat]
        # h: [N, output_feat]
        # M是alpha
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        # p是Lp normalize标准化中的p，dim是the dimension to reduce. Default: 1
        # z: [N, output_feat]
        z = F.normalize(h, p=2, dim=1)
        # 功能：将某一个维度除以那个维度对应的范数(默认是2范数)
        # F.normalize(data, p=2/1, dim=0/1/-1) 将某一个维度除以那个维度对应的范数(默认是2范数)
        #
        #         data:输入的数据（tensor）
        #
        #         p:L2/L1_norm运算
        #是除以该
        #         #         dim:0表示按列操作，则每列都列下平方和的开方；1表示按行操作，则每行都是除以该行下所有元素平方和的开方
        # decoder, A: [N, N]
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        x = torch.matmul(Z, Z.t())
        A_pred = torch.sigmoid(x-1/x)
        return A_pred
    def _decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
def GetPrearr(x, num_cluster):
    matrix = np.zeros((len(x), num_cluster))
    for i in range(len(x)):
        for j in range(num_cluster):
            matrix[i][j] = 0
            matrix[i][x[i]] = 1
    return matrix


def Modula(array, cluster):
    m = sum(sum(array)) / 2
    k1 = np.sum(array, axis=1)
    k2 = k1.reshape(k1.shape[0], 1)
    k1k2 = k1 * k2
    Eij = k1k2 / (2 * m)
    B = array - Eij
    node_cluster = np.dot(cluster, np.transpose(cluster))
    results = np.dot(B, node_cluster)
    sum_results = np.trace(results)
    modul = sum_results / (2 * m)
    return modul
