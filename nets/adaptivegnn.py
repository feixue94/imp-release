# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> adaptivegnn
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   18/03/2022 14:49
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


# from nets.basic_layers import attention, MLP
def MLP_IN(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
                # layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class AdaMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, M=None):
        '''
        :param query: [B, D, N]
        :param key: [B, D, M]
        :param value: [B, D, M]
        :param M: [B, N, M]
        :return:
        '''
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]  # [B, D, NH, N]
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5

        if M is not None:
            scores = scores * M[:, None, :].expand_as(scores)
        prob = F.softmax(scores, dim=-1)
        x = torch.einsum('bhnm,bdhm->bdhn', prob, value)
        self.prob = torch.mean(prob, dim=1)

        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AdaAttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = AdaMultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP_IN([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, M=None):
        message = self.attn(x, source, source, M=M)
        return self.mlp(torch.cat([x, message], dim=1))


class AdaAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, pooling_sizes: list = None, keep_ratios: list = None):
        super().__init__()
        self.layers = nn.ModuleList([
            AdaAttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

        if pooling_sizes is None and keep_ratios is None:
            self.use_ada = False
        else:
            self.pooling_sizes = pooling_sizes
            self.keep_ratios = keep_ratios

            if self.pooling_sizes is not None:
                self.pooling_values = self.pooling_sizes
                self.pool_fun = self.update_connections_by_pooling_size
            elif self.keep_ratios is not None:
                self.pooling_values = self.keep_ratios
                self.pool_fun = self.update_connections_by_ratio

    def forward(self, desc0, desc1, require_prob=False):
        bs = desc0.shape[0]
        N = desc0.shape[2]
        M = desc1.shape[2]
        M00 = torch.ones(size=(bs, N, N), device=desc0.device).float()
        M01 = torch.ones(size=(bs, N, M), device=desc0.device).float()
        M11 = torch.ones(size=(bs, M, M), device=desc0.device).float()
        M10 = torch.ones(size=(bs, M, N), device=desc0.device).float()
        P00 = None
        P01 = None
        P11 = None
        P10 = None

        for i, (layer, name, pool_value) in enumerate(zip(self.layers, self.names, self.pooling_values)):
            if name == 'cross':
                M01 = self.pool_fun(M01, P01, pool_value)
                delta0 = layer(desc0, desc1, M=M01)
                P01 = layer.attn.prob

                M10 = self.pool_fun(M10, P10, pool_value)
                delta1 = layer(desc1, desc0, M=M10)
                P10 = layer.attn.prob
            else:
                M00 = self.pool_fun(M00, P00, pool_value)
                delta0 = layer(desc0, desc0, M=M00)
                P00 = layer.attn.prob

                M11 = self.pool_fun(M11, P11, pool_value)
                delta1 = layer(desc1, desc1, M=M11)
                P11 = layer.attn.prob

            print(i // 2 + 1, name, pool_value,
                  # torch.cumsum(M00 > 0, dim=-1), torch.cumsum(M01 > 0, dim=-1),
                  # torch.cumsum(M10 > 0, dim=-1), torch.cumsum(M11 > 0, dim=-1),
                  )

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

        return desc0, desc1

    def update_connections_by_pooling_size(self, last_connections, last_prob, pool_size):
        if last_connections is None:
            return None
        if last_prob is None or pool_size is None:
            return last_connections
        if pool_size == 1:
            return last_connections

        # bs = last_connections.shape[0]
        current_connections = torch.zeros_like(last_connections)
        k = int(torch.sum(last_connections[0], dim=-1)[0].item() // pool_size)
        # print('k: ', k, last_connections.dtype, last_prob.dtype)
        values, idxs = last_prob.topk(k=k, largest=True, dim=2)
        current_connections = current_connections.scatter(2, idxs, values)
        current_connections[current_connections > 0] = 1
        return current_connections

    def update_connections_by_ratio(self, last_connections, last_prob, ratio):
        if last_connections is None:
            return None
        if last_prob is None or ratio is None:
            return last_connections
        if ratio == 1:
            return last_connections

        with torch.no_grad():
            bs = last_connections.shape[0]
            current_connections = torch.zeros_like(last_connections)
            # idxs = torch.argsort(last_prob, descending=True, dim=2)
            # print('idxs: ', idxs.shape)
            # sorted_prob = torch.zeros_like(last_prob)
            sorted_prob, idxs = torch.sort(last_prob, dim=2, descending=True)
            print('prob: ', sorted_prob.shape, idxs.shape)

            acc_prob = torch.cumsum(sorted_prob, dim=2)
            # acc_prob[(acc_prob < ratio)] += 10
            # print('acc_prob: ', acc_prob.shape)
            # values, idxs2 = torch.min(acc_prob, dim=2)
            current_connections = current_connections.scatter(2, idxs(acc_prob <= ratio),
                                                              sorted_prob(acc_prob <= ratio))
            current_connections[current_connections > 0] = 1
            # for bid in range(bs):
            #     current_connections[bid, idxs[:idxs2[bid] + 1]] = 1

    def gen_grid(self, nx: int, ny: int):
        '''
        :param nx:
        :param ny:
        :return: out[0] = [0, 0, ,,,, 1, 1,...,ny], out[1] = [0, 1, 2, ..., nx, 0, 1,...,nx, ]
        '''
        x1 = torch.arange(0, nx)
        y1 = torch.arange(0, ny)
        H1, W1 = len(y1), len(x1)
        x1 = x1[None, :].expand(H1, W1).reshape(-1)
        y1 = y1[:, None].expand(H1, W1).reshape(-1)

        return torch.vstack([y1, x1])


if __name__ == '__main__':
    bs = 4
    n = 1024
    d = 256
    desc0 = torch.rand(bs, d, n).float().cuda()
    desc1 = torch.rand(bs, d, n).float().cuda()
    layer_names = ['self', 'cross'] * 9
    pool_sizes = [1, 1] * 5 + [2, 2] * 4
    keep_ratios = [1, 1] * 5 + [0.5, 0.5] * 4
    net = AdaAttentionalGNN(feature_dim=d, layer_names=layer_names, pooling_sizes=pool_sizes).cuda()
    # net = AdaAttentionalGNN(feature_dim=d, layer_names=layer_names, keep_ratios=keep_ratios).cuda()

    for i in range(1000):
        print('it: ', i)
        out = net(desc0, desc1)
        exit(0)
        # del out
        # torch.cuda.memory_cached()
