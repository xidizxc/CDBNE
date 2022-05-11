import torch
import torch.nn as nn
import torch.nn.functional as F


class GATELayer(nn.Module):
    """
    Simple GATE layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        jihuo = 1.414
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # Xavier均匀分布
        # xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)，
        # 这里有一个gain，增益的大小是依据激活函数类型来设定
        # eg：nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
        #
        # PS：上述初始化方法，也称为Glorot initialization
        nn.init.xavier_uniform_(self.W.data, gain=jihuo)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        # print(self.a_self.data)
        nn.init.xavier_uniform_(self.a_self.data, gain=jihuo)
        # print(self.a_self.data)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=jihuo)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # print(self.leakyrelu)

    def forward(self, input, adj, M, concat=True):
        # torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
        h = torch.mm(input, self.W)
        # x: [samples_cnt=N, input_feat]
        # w: [input_feat, output_feat]
        # h: [N, output_feat]
        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        # >>> a
        # tensor([[1],
        #         [2],
        #         [3]])
        # >>> torch.transpose(a, 0, 1)
        # tensor([[1, 2, 3]])
        # >>> a+torch.transpose(a, 0, 1)
        # tensor([[2, 3, 4],
        #         [3, 4, 5],
        #         [4, 5, 6]])
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)  # [N, N]
        # [N, N]*[N, N]=>[N, N]
        attn_dense = torch.mul(attn_dense, M)
        # torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj)  # [N, N] #生成与邻接矩阵adj尺寸相似的矩阵
        # torch.where: Return a tensor of elements selected from either x or y, depending on condition
        # torch.where(condition, x, y) → Tensor, xi if condition else yi 如果adj>0 那么adj=attn_dense
        adj = torch.where(adj > 0, attn_dense, zero_vec)  # [N, N]
        # 对每一行的样本所有邻居softmax
        attention = F.softmax(adj, dim=1)  # [N, N]
        # attention: [N, N]
        # h: [N, output_feat]
        h_prime = torch.matmul(attention, h)  # N, output_feat

        if concat:
            # torch.nn.function.elu: Applies element-wise, ELU(x)=max(0,x)+min(0,α∗(exp(x)−1)) .
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )