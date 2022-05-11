import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像"""
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    # 遍历所有样本
    # data.shape[0]为节点数
    # for i in range(data.shape[0]):
    #     # 在图中为每个数据点画出标签
    #    plt.text(data[i, 0], data[i, 1], str("."), color=plt.cm.Set1(label[i] / 10),
    #              fontdict={'weight': 'bold', 'size': 30})
    # plt.xticks()  # 指定坐标的刻度
    # plt.yticks()
    # plt.title(title, fontsize=14)
    # 返回值
    return fig

