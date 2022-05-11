import argparse
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import utils
from model import GATE
from model import Modula
from evaluation import eva
from sklearn.manifold import TSNE
import pandas as pd
from torch.nn import init
import matplotlib.pyplot as plt

# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)
#
# net = nn.Sequential(
#     nn.Linear(num_inputs, 1)
#     # 此处还可以传入其他层
#     )
# print(net)
# print(net[0])
from visiuation import plot_embedding


def maxmetr(acc, nmi, ari, f1):
    print(f"max :acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
    return


class CDBNE(nn.Module):
    def __init__(self, num_features, middle_size, representation_size, alpha, clusters_number, v=1):
        super(CDBNE, self).__init__()
        self.clusters_number = clusters_number
        self.v = v

        # get model
        self.gate = GATE(num_features, middle_size, representation_size, alpha)
        self.gate.load_state_dict(torch.load(args.pathes, map_location='cpu'))

        # cluster layer   # cluster layer，簇头embed
        self.cluster_layer = Parameter(torch.Tensor(clusters_number, representation_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        # 得到reconstruct的邻接和[N, feat_size]的节点embedding Z
        A_pred, z_embedding = self.gate(x, adj, M)
        q = self.modularity(z_embedding)
        return A_pred, z_embedding, q

    def modularity(self, z_embedding):
        dist = torch.sum(torch.pow(z_embedding.unsqueeze(1) - self.cluster_layer, 2), 2)
        q = 1.0 / (1.0 + dist / self.v) ** ((self.v + 1.0) / 2.0)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_fenbu(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def trainer(shujuji):
    model = CDBNE(num_features=args.input_dim, middle_size=args.middle_size,
                  representation_size=args.representation_size, alpha=args.alpha, clusters_number=args.n_clusters).to(
        device)
    # print(model)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer1 = optim.SGD(net.parameters(), lr=0.03)
    # data process
    shujuji = utils.data_preprocessing(shujuji)
    adj = shujuji.adj.to(device)
    adj_label = shujuji.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data label
    data = torch.Tensor(shujuji.x).to(device)
    y = shujuji.y.cpu().numpy()
    # y的尺寸为2708 [3 4 4 ... 3 3 3]

    with torch.no_grad():
        # 相当epoch的模型做一eval
        _, z_embedding = model.gate(data, adj, M)

    # get kmeans and pre result
    # 这里是用pre结果来初始化kmean中心
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    kmeans.fit(z_embedding.data.cpu())
    # vis.plot_embedding(z_embedding.data.cpu())
    y_pred = kmeans.fit_predict(z_embedding.data.cpu().numpy())  # 得到label
    # y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    # plt.scatter(z_embedding.data.cpu().numpy()[:, 0], z_embedding.data.cpu().numpy()[:, 1], c=y_pred)
    # plt.show()
    # y_pred = DBSCAN().fit_predict(z_embedding.data.cpu().numpy())
    # eva(y, y_pred, 'pre')
    # model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # print(kmeans.cluster_centers_)
    ADj = np.array(utils.data_preprocessing(shujuji).adj)  # 得到正常邻接矩阵
    listacc = []
    listnmi = []
    listari = []
    listf1 = []

    # i = 1
    for epoch in range(args.max_epoch):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z_embedding, Q = model(data, adj, M)
            q = Q.detach().data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(y, q, epoch)
            listacc.append(acc)
            listnmi.append(nmi)
            listari.append(ari)
            listf1.append(f1)

        A_pred, z_embedding, q = model(data, adj, M)
        # 输入 2708*16
        z_embeddin = TSNE(n_components=2).fit_transform(z_embedding.data.cpu().numpy())
        '''分隔符'''  # 这里是聚类可视化
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        # plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=y_pred, s=20)
        color_set = ('gray', 'green', 'red', 'cyan', 'magenta', 'yellow', 'blue')
        color_list = [color_set[int(label)] for label in y_pred]
        plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=color_list, s=30)
        plt.show()
        '''分隔符'''
        # fig = plot_embedding(z_embeddin, y, 'julei')
        # 显示图像
        # plt.show()
        p = target_fenbu(Q.detach())
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        MU_loss = Modula(adj.detach().numpy(), A_pred.detach().numpy())

        loss = 0.001 * kl_loss + re_loss - 0.1 * MU_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # optimizer1.step()
    print(f"max :acc {max(listacc):.4f}, nmi {max(listnmi):.4f}, ari {max(listari):.4f}, f1 {max(listf1):.4f}")


def GetPrearr(x, num_cluster):
    matrix = np.zeros((len(x), num_cluster))
    for i in range(len(x)):
        for j in range(num_cluster):
            matrix[i][j] = 0
            matrix[i][x[i]] = 1
    return matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--alpha1', type=float, default=1, help="p and q")
    parser.add_argument('--name', type=str, default='Cora')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--middle_size', default=256, type=int)
    parser.add_argument('--representation_size', default=16, type=int)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    datasets = utils.get_dataset(args.name)
    shujuji = datasets[0]

    if args.name == 'Cora':
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 7
        args.epoch = 1
        args.v = 1
    else:
        args.k = None

    args.pathes = f'./pre/preCDBNE_{args.name}_{args.epoch}.pkl'
    args.input_dim = shujuji.num_features

    print(args)
    trainer(shujuji)
