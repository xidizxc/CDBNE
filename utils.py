import numpy as np
import torch
from sklearn.preprocessing import normalize

from torch_geometric.datasets import Planetoid


def get_dataset(dataset):
    datasets = Planetoid(root='./dataset', name=dataset)
    # print("训练集节点数量",sum(datasets.data.train_mask))
    return datasets

def data_preprocessing(dataset):
    # 其实就是用边构建邻接矩阵，参考 https://pytorch.apachecn.org/docs/1.0/torch_tensors.html
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    # torch.eye: 返回二维张量，对角线上是1，其它地方是0.
    # 给邻接矩阵加上节点到自己的边
    dataset.adj += torch.eye(dataset.x.shape[0])
    # 每个元素除以每行的l1范数，即每行元素和，如果是l2就是除以每行样本的l2范数
    # 这里的adj就是论文中的 transition matrix B_{ij}=1/d_i if e_{ij} \in E
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    # M就是论文中的proximity matrix M
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


