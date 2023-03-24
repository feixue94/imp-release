# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp-release -> layers
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   24/03/2023 10:51
=================================================='''
import torch
import torch.nn as nn
from copy import deepcopy

eps = 1e-8


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def dual_softmax(M, dustbin):
    M = torch.cat([M, dustbin.expand([M.shape[0], M.shape[1], 1])], dim=-1)
    M = torch.cat([M, dustbin.expand([M.shape[0], 1, M.shape[2]])], dim=-2)
    score = torch.log_softmax(M, dim=-1) + torch.log_softmax(M, dim=1)
    return torch.exp(score)


def sinkhorn(M, r, c, iteration):
    p = torch.softmax(M, dim=-1)
    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iteration):
        u = r / ((p * v.unsqueeze(-2)).sum(-1) + eps)
        v = c / ((p * u.unsqueeze(-1)).sum(-2) + eps)
    p = p * u.unsqueeze(-1) * v.unsqueeze(-2)
    return p


def sink_algorithm(M, dustbin, iteration):
    M = torch.cat([M, dustbin.expand([M.shape[0], M.shape[1], 1])], dim=-1)
    M = torch.cat([M, dustbin.expand([M.shape[0], 1, M.shape[2]])], dim=-2)
    r = torch.ones([M.shape[0], M.shape[1] - 1], device='cuda')
    r = torch.cat([r, torch.ones([M.shape[0], 1], device='cuda') * M.shape[1]], dim=-1)
    c = torch.ones([M.shape[0], M.shape[2] - 1], device='cuda')
    c = torch.cat([c, torch.ones([M.shape[0], 1], device='cuda') * M.shape[2]], dim=-1)
    p = sinkhorn(M, r, c, iteration)
    return p


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def MLP(channels: list, ac_fn='relu', norm_fn='bn'):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if norm_fn == 'in':
                layers.append(nn.InstanceNorm1d(channels[i], eps=1e-3))
            elif norm_fn == 'bn':
                layers.append(nn.BatchNorm1d(channels[i], eps=1e-3))
            if ac_fn == 'relu':
                layers.append(nn.ReLU())
            elif ac_fn == 'gelu':
                layers.append(nn.GELU())
            elif ac_fn == 'lrelu':
                layers.append(nn.LeakyReLU(negative_slope=0.1))
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers, ac_fn='relu', norm_fn='bn'):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim], ac_fn=ac_fn, norm_fn=norm_fn)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]  # [B, 2, N] + [B, 1, N]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)

        self.prob = torch.mean(prob, dim=1, keepdim=False)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1, require_prob=False):
        probs = {}
        desc0s = []
        desc1s = []

        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            # layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0 = layer(desc0, src0)
            prob0 = layer.attn.prob
            delta1 = layer(desc1, src1)
            prob1 = layer.attn.prob
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

            probs[i] = {
                'name': name,
                'prob0': prob0,
                'prob1': prob1,
            }
            if name == 'cross':
                desc0s.append(desc0)
                desc1s.append(desc1)

        if require_prob:
            return desc0s, desc1s, probs
        else:
            return desc0s, desc1s


class SharedAttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, sharing_attention: bool = False, ac_fn: str = 'relu',
                 norm_fn: str = 'bn'):
        super().__init__()
        self.sharing_attention = sharing_attention
        self.feature_dim = feature_dim
        if not sharing_attention:
            self.attn = MultiHeadedAttention(num_heads, feature_dim)
            self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim], ac_fn=ac_fn, norm_fn=norm_fn)
            nn.init.constant_(self.mlp[-1].bias, 0.0)
        else:
            self.dim = feature_dim // num_heads
            self.num_heads = num_heads
            self.proj = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
            self.merge = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
            self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim], ac_fn=ac_fn, norm_fn=norm_fn)
            nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, prob=None, M=None):
        """
        :param x: [B, C, N]
        :param source: [B, C, N]
        :param prob: [B, C, H, N]
        :return: [B, C, N]
        """
        if not self.sharing_attention:
            message = self.attn(x, source, source, M=M)
            self.prob = self.attn.prob
            y = torch.cat([x, message], dim=1)
        else:
            batch_dim = x.size(0)
            value = self.proj(source).view(batch_dim, self.dim, self.num_heads, -1)
            message = torch.einsum('bhnm,bdhm->bdhn', prob, value)
            message = self.merge(message.contiguous().view(batch_dim, self.dim * self.num_heads, -1))
            self.prob = prob
            y = torch.cat([x, message], dim=1)

            # out = self.mlp(torch.cat([x, message], dim=1))

        # print('after mlp: ', self.sharing_attention, out[0, :-3, 0:5], x[0, :-3, 0:5], message[0, :-3, 0:5])
        # print('mlp weights: ', self.mlp.state_dict())

        '''
        if M is None:
            return self.mlp(y)
        else:
            with torch.no_grad():
                sids0 = torch.where(torch.sum(M, dim=-1) > 0)[1]
            out = torch.zeros(size=(y.shape[0], self.feature_dim, y.shape[-1]), dtype=y.dtype, device=y.device)
            # sids1 = torch.where(torch.sum(M, dim=-1) == 0)[1]
            out[:, :, sids0] = self.mlp(y[:, :, sids0])
            # out[:, :, sids1] = self.mlp(y[:, :, sids1])
            return out
        '''
        return self.mlp(y)


class SAGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list,
                 sharing_layers: list = None, ac_fn='relu', norm_fn='bn'):
        super().__init__()
        if sharing_layers is None:
            self.sharing_layers = [False for i in range(len(layer_names))]
        else:
            self.sharing_layers = sharing_layers
        self.layers = nn.ModuleList([
            SharedAttentionalPropagation(num_heads=4, feature_dim=feature_dim, sharing_attention=self.sharing_layers[i],
                                         ac_fn=ac_fn, norm_fn=norm_fn)
            for i in range(len(layer_names))
        ])
        self.names = layer_names
