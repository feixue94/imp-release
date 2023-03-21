# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> layers
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   16/05/2022 11:04
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nets.fast_kmeans import KMeans
from copy import deepcopy
import numpy as np
from einops import rearrange

eps = 1e-8


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


# def MLP(channels: list, do_bn=True, ac_fn='relu', norm_fn='bn'):
#     """ Multi-layer perceptron """
#     n = len(channels)
#     layers = []
#     for i in range(1, n):
#         if norm_fn == 'ln':  # pre-normalization for ln
#             if i == 1:  # only for the first the layer as vit
#                 layers.append(nn.LayerNorm(channels[i - 1]))
#         layers.append(
#             nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
#         if i < (n - 1):
#             if norm_fn == 'in':
#                 layers.append(nn.InstanceNorm1d(channels[i], eps=1e-3))
#             elif norm_fn == 'bn':
#                 layers.append(nn.BatchNorm1d(channels[i], eps=1e-3))
#             if ac_fn == 'relu':
#                 layers.append(nn.ReLU())
#             elif ac_fn == 'gelu':
#                 layers.append(nn.GELU())
#             elif ac_fn == 'lrelu':
#                 layers.append(nn.LeakyReLU(negative_slope=0.1))
#             if norm_fn == 'ln':
#                 layers.append(nn.LayerNorm(channels[i]))
#     return nn.Sequential(*layers)

class SampleLN(nn.Module):
    def __init__(self, dim, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        if len(x.shape) == 2:  # [B, D]
            return self.norm(x)
        elif len(x.shape) == 3:  # [B, D, N]
            b = x.shape[0]
            y = self.norm(rearrange(x, 'b d n -> (b n) d'))
            return rearrange(y, '(b n) d -> b d n', b=b)


def MLP(channels: list, do_bn=True, ac_fn='relu', norm_fn='bn'):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        if norm_fn == 'ln':
            if i == 1:
                layers.append(SampleLN(channels[i - 1]))
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


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers, ac_fn='relu', norm_fn='bn'):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim], ac_fn=ac_fn, norm_fn=norm_fn)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]  # [B, 2, N] + [B, 1, N]
        return self.encoder(torch.cat(inputs, dim=1))


class MultiHeadedAttention(nn.Module):
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

        # print('query: ', query[0, 0, 0, :5])
        # print('key: ', key[0, 0, 0, :5])
        # print('value: ', value[0, 0, 0, :5])
        # print('scores: ', scores[0, 0, :5, :5])

        if M is not None:
            # print('M: ', scores.shape, M.shape, torch.sum(M, dim=2))
            # scores = scores * M[:, None, :, :].expand_as(scores)
            # with torch.no_grad():
            mask = (1 - M[:, None, :, :]).repeat(1, scores.shape[1], 1, 1).bool()  # [B, H, N, M]
            scores = scores.masked_fill(mask, -torch.finfo(scores.dtype).max)
            prob = F.softmax(scores, dim=-1)  # * (~mask).float()  # * mask.float()
            # prob = prob * ones
            # prob = prob.masked_fill(mask, 0.)

            # print('before mask 0: ', torch.sum(prob[0, 0], dim=-1)[:5])
            # prob[mask] = 0
            # print('after mask 0: ', torch.sum(prob[0, 0], dim=-1)[:5])

            # print('using mask...')
            # print('mask: ', torch.where(M > 0)[1], torch.where(M > 0)[2])
            #
            # print('prob w/m: ', prob[:, :, 3, 3], torch.min(prob), torch.max(prob))
            # sids0 = torch.unique(torch.where(M > 0)[1])
            # sids1 = torch.unique(torch.where(M > 0)[2])
            # print('sids: ', sids0.shape, sids1.shape)
            # s = scores[:, :, sids0, :][:, :, :, sids1]
            # p = F.softmax(s, dim=-1)
            # print('prob w/o m: ', p[:, :, 0, 0])
            # print(M.shape, scores.shape, prob.shape, p.shape, torch.min(p), torch.max(p))
            # print('max: ', torch.sum(prob, dim=1).sum(dim=1)[:, sids1], torch.sum(p, dim=1).sum(dim=1))
            # exit(0)
        else:
            prob = F.softmax(scores, dim=-1)

        x = torch.einsum('bhnm,bdhm->bdhn', prob, value)
        self.prob = prob

        out = self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))

        return out


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, ac_fn='relu', norm_fn='bn'):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim], ac_fn=ac_fn, norm_fn=norm_fn)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, M=None):
        message = self.attn(x, source, source, M=M)
        self.prob = self.attn.prob

        out = self.mlp(torch.cat([x, message], dim=1))
        # if M is not None:
        #     mask = torch.sum(M, dim=-1) == 0
        #     mask = mask[:, None, :].repeat(1, out.shape[1], 1)
        #     out[mask] = 0
        return out


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


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, ac_fn='relu', batch_fn='bn'):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4, ac_fn=ac_fn, norm_fn=batch_fn)
            for _ in range(len(layer_names))
        ])
        self.names = layer_names

    def forward(self, desc0, desc1):
        all_desc0s = []
        all_desc1s = []
        bs = desc0.shape[0]
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc1, desc0], dim=0), None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]
            else:
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc0, desc1], dim=0), None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

            # desc0 = delta0
            # desc1 = delta1

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)
        return all_desc0s, all_desc1s, []

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        bs = desc0.shape[0]
        layer = self.layers[layer_i]
        name = self.names[layer_i]
        if name == 'cross':
            delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc1, desc0], dim=0),
                          None if M0 is None or M1 is None else torch.cat([M0, M1], dim=0))
            delta0 = delta[:bs]
            delta1 = delta[bs:]
        else:
            delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc0, desc1], dim=0),
                          None if M0 is None or M1 is None else torch.cat([M0, M1], dim=0))
            delta0 = delta[:bs]
            delta1 = delta[bs:]

        # return delta0, delta1
        return desc0 + delta0, desc1 + delta1


class KMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, weight=None, M=None):
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

        if weight is not None:
            scores = scores * weight[:, None, None, :].expand_as(scores)

        if M is not None:
            # print('M: ', scores.shape, M.shape, torch.sum(M, dim=2))
            scores = scores * M[:, None, :].expand_as(scores)
        prob = F.softmax(scores, dim=-1)
        x = torch.einsum('bhnm,bdhm->bdhn', prob, value)
        self.prob = torch.mean(prob, dim=1)

        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class KAttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = KMultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, weight=None, M=None):
        message = self.attn(x, source, source, weight, M=M)
        return self.mlp(torch.cat([x, message], dim=1))


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinAttention(nn.Module):
    def __init__(self, seq_length: int, num_heads: int, d_model: int, k: int, share_kv=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.share_kv = share_kv
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.to_q = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.to_k = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_length, k)))

        if not self.share_kv:
            self.to_v = nn.Conv1d(d_model, d_model, kernel_size=1)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_length, k)))

    def forward(self, query, key, value, M=None, **kwargs):
        '''
        :param query: [B, D, N]
        :param key: [B, D, M]
        :param value: [B, D, M]
        :param M: [B, N, M]
        :return:
        '''

        batch_dim = query.size(0)
        query = self.to_q(query)
        key = self.to_k(key)
        value = self.to_v(value) if not self.share_kv else key

        proj_seq = lambda args: torch.einsum('bdn,nk->bdk', *args)
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)
        key, value = map(proj_seq, zip((key, value), kv_projs))  # [B, ]

        query = query.view(batch_dim, self.dim, self.num_heads, -1)
        key = key.view(batch_dim, self.dim, self.num_heads, -1)
        value = value.view(batch_dim, self.dim, self.num_heads, -1)

        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhk->bhnk', query, key) / dim ** .5

        # if M is not None:
        #     scores = scores * M[:, None, :].expand_as(scores)
        prob = F.softmax(scores, dim=-1)
        x = torch.einsum('bhnk,bdhk->bdhn', prob, value)
        self.prob = torch.mean(prob, dim=1)

        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class SCLAttention(nn.Module):
    def __init__(self, seq_length: int, num_heads: int, d_model: int, k: int, share_kv=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.share_kv = share_kv
        self.dim = d_model // num_heads
        self.num_heads = num_heads

        # For sequence 1
        self.to_q1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.to_k1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj_k1 = nn.Parameter(init_(torch.zeros(seq_length, k)))
        if not self.share_kv:
            self.to_v1 = nn.Conv1d(d_model, d_model, kernel_size=1)
            self.proj_v1 = nn.Parameter(init_(torch.zeros(seq_length, k)))
        self.to_out1 = nn.Conv1d(d_model, d_model, kernel_size=1)

        # For sequence 2
        self.to_q2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.to_k2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj_k2 = nn.Parameter(init_(torch.zeros(seq_length, k)))
        if not self.share_kv:
            self.to_v2 = nn.Conv1d(d_model, d_model, kernel_size=1)
            self.proj_v2 = nn.Parameter(init_(torch.zeros(seq_length, k)))
        self.to_out2 = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.mlp = MLP([d_model * 3, d_model * 2, d_model])

    def attention(self, query, key, value):
        batch_dim = query.size(0)
        query = query.view(batch_dim, self.dim, self.num_heads, -1)
        key = key.view(batch_dim, self.dim, self.num_heads, -1)
        value = value.view(batch_dim, self.dim, self.num_heads, -1)
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhk->bhnk', query, key) / dim ** .5

        prob = F.softmax(scores, dim=-1)
        x = torch.einsum('bhnk,bdhk->bdhn', prob, value)

        return x.contiguous().view(batch_dim, self.dim * self.num_heads, -1)

    def forward(self, x1, x2, **kwargs):
        '''
        :param query: [B, D, N]
        :param key: [B, D, M]
        :param value: [B, D, M]
        :param M: [B, N, M]
        :return:
        '''
        proj_seq = lambda args: torch.einsum('bdn,nk->bdk', *args)

        query1 = self.to_q1(x1)
        key1 = self.to_k1(x2)
        value1 = self.to_v1(x2) if not self.share_kv else key1
        kv_projs1 = (self.proj_k1, self.proj_v1 if not self.share_kv else self.proj_k1)
        key1, value1 = map(proj_seq, zip((key1, value1), kv_projs1))  # [B, ]

        query2 = self.to_q2(x2)
        key2 = self.to_k2(x1)
        value2 = self.to_v2(x1) if not self.share_kv else key2
        proj_seq = lambda args: torch.einsum('bdn,nk->bdk', *args)
        kv_projs2 = (self.proj_k2, self.proj_v2 if not self.share_kv2 else self.proj_k2)
        key2, value2 = map(proj_seq, zip((key2, value2), kv_projs2))  # [B, ]

        x11 = self.attention(query=query1, key=key1, value=value1)
        x22 = self.attention(query=query2, key=key2, value=value2)
        x12 = self.attention(query=query1, key=key2, value=value2)
        x21 = self.attention(query=query2, key=key1, value=value1)

        x1_out = self.mlp1(torch.cat([x11, x12], dim=1))
        x2_out = self.mlp2(torch.cat([x22, x21], dim=1))

        return x1_out, x2_out


class LinAttentionalPropagation(nn.Module):
    def __init__(self, seq_length: int, feature_dim: int, num_heads: int, k: int, share_kv: bool = True):
        super().__init__()
        self.attn = LinAttention(seq_length=seq_length, d_model=feature_dim, num_heads=num_heads, k=128,
                                 share_kv=share_kv)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, M=None):
        message = self.attn(x, source, source, M=M)
        return self.mlp(torch.cat([x, message], dim=1))


class MultiLinAGNN(nn.Module):
    def __init__(self, seq_length: int, feature_dim: int, layer_names: list, k: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([
            LinAttentionalPropagation(seq_length=seq_length, feature_dim=feature_dim, num_heads=4, k=k, share_kv=True)
            for _ in range(len(layer_names))
        ])
        self.names = layer_names

    def forward(self, desc0, desc1):
        all_desc0s = []
        all_desc1s = []
        bs = desc0.shape[0]
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc1, desc0], dim=0), None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]
            else:
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc0, desc1], dim=0), None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)
        return all_desc0s, all_desc1s, []

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        bs = desc0.shape[0]
        layer = self.layers[layer_i]
        name = self.names[layer_i]
        if name == 'cross':
            delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc1, desc0], dim=0),
                          None if M0 is None or M1 is None else torch.cat([M0, M1], dim=0))
            delta0 = delta[:bs]
            delta1 = delta[bs:]
        else:
            delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc0, desc1], dim=0),
                          None if M0 is None or M1 is None else torch.cat([M0, M1], dim=0))
            delta0 = delta[:bs]
            delta1 = delta[bs:]

        return desc0 + delta0, desc1 + delta1


class KMultiLinAGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, k: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            KAttentionalPropagation(num_heads=4, feature_dim=feature_dim)
            for _ in range(len(layer_names))
        ])
        self.names = layer_names
        self.k = k
        self.KMeans = KMeans(n_clusters=k, max_iter=50)
        self.minibatch = None

    def kmeans(self, X, centroids=None, minibatch=None):
        with torch.no_grad():
            cluster_ids, cluster_counts, cluster_centers = self.KMeans.batch_kmeans(
                X=X.transpose(1, 2),
                centroids=centroids.transpose(1, 2) if centroids is not None else None,
                minibatch=minibatch)
        return cluster_ids, cluster_counts, cluster_centers.transpose(1, 2)

    def forward(self, desc0, desc1):
        all_desc0s = []
        all_desc1s = []
        bs = desc0.shape[0]
        cluster_centers = None
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            # print(i, name)
            if name == 'cross':
                cluster_ids, cluster_counts, cluster_centers = self.kmeans(X=torch.cat([desc1, desc0], dim=0),
                                                                           centroids=cluster_centers,
                                                                           minibatch=self.minibatch)
                weight = cluster_counts.float()

                delta = layer(torch.cat([desc0, desc1], dim=0), cluster_centers, weight, None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]
            else:  # self
                cluster_ids, cluster_counts, cluster_centers = self.kmeans(X=torch.cat([desc0, desc1], dim=0),
                                                                           centroids=cluster_centers,
                                                                           minibatch=self.minibatch)
                weight = cluster_counts.float()
                # print(cluster_ids.shape, weight.shape, weight, ids.shape)
                delta = layer(torch.cat([desc0, desc1], dim=0), cluster_centers, weight, None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)
        return all_desc0s, all_desc1s, []

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        bs = desc0.shape[0]
        layer = self.layers[layer_i]
        name = self.names[layer_i]
        cluster_centers = None
        if name == 'cross':
            cluster_ids, cluster_counts, cluster_centers = self.kmeans(X=torch.cat([desc1, desc0], dim=0),
                                                                       centroids=cluster_centers)
            weight = cluster_counts.float()
            delta = layer(torch.cat([desc0, desc1], dim=0), cluster_centers, weight,
                          None if M0 is None or M1 is None else torch.cat([M0, M1], dim=0))
            delta0 = delta[:bs]
            delta1 = delta[bs:]
        else:
            cluster_ids, cluster_counts, cluster_centers = self.kmeans(X=torch.cat([desc0, desc1], dim=0),
                                                                       centroids=cluster_centers)
            weight = cluster_counts.float()
            delta = layer(torch.cat([desc0, desc1], dim=0), cluster_centers, weight,
                          None if M0 is None or M1 is None else torch.cat([M0, M1], dim=0))
            delta0 = delta[:bs]
            delta1 = delta[bs:]

        return desc0 + delta0, desc1 + delta1


class SMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, topk: int = 128):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.topk = topk

    def forward(self, query, key, value, M=None):
        '''
        :param query: [B, D, N]
        :param key: [B, D, M]
        :param value: [B, D, M]
        :param M: [B, N, M]
        :return:
        '''
        batch_dim = query.size(0)
        query, key, value = [l(x)
                             for l, x in zip(self.proj, (query, key, value))]  # [B, D, NH, N]

        KV = torch.cat([key, value], dim=0)
        _, _, Vh = torch.linalg.svd(KV)
        new_KV = KV @ Vh[:, :, :self.topk]
        key = new_KV[:batch_dim]
        value = new_KV[batch_dim:]

        query = query.view(batch_dim, self.dim, self.num_heads, -1)
        key = key.view(batch_dim, self.dim, self.num_heads, -1)
        value = value.view(batch_dim, self.dim, self.num_heads, -1)

        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5

        if M is not None:
            # print('M: ', scores.shape, M.shape, torch.sum(M, dim=2))
            scores = scores * M[:, None, :].expand_as(scores)
        prob = F.softmax(scores, dim=-1)
        x = torch.einsum('bhnm,bdhm->bdhn', prob, value)
        self.prob = torch.mean(prob, dim=1)

        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class SAttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, topk: int = 128):
        super().__init__()
        self.attn = SMultiHeadedAttention(num_heads, feature_dim, topk=topk)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, M=None):
        message = self.attn(x, source, source, M=M)
        return self.mlp(torch.cat([x, message], dim=1))


class SAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, topk: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([
            SAttentionalPropagation(feature_dim, 4, topk=topk)
            for _ in range(len(layer_names))
        ])
        self.names = layer_names

    def forward(self, desc0, desc1):
        all_desc0s = []
        all_desc1s = []
        bs = desc0.shape[0]
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc1, desc0], dim=0), None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]
            else:
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc0, desc1], dim=0), None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)
        return all_desc0s, all_desc1s, []


class HAttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, pool_size: int = 0):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads=num_heads, d_model=feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

        if pool_size > 0:
            self.pooling = nn.MaxPool1d(kernel_size=pool_size)
        else:
            self.pooling = None

    def forward(self, x, source, M=None):
        if self.pooling is not None:
            source = self.pooling(source)
        message = self.attn(x, source, source, M=M)
        return self.mlp(torch.cat([x, message], dim=1))


class HAGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, pool_sizes: list = None):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(num_heads=4, feature_dim=feature_dim)
            for i in range(len(layer_names))
        ])

        self.names = layer_names
        self.pool_sizes = pool_sizes

    def forward(self, desc0, desc1):
        all_desc0s = []
        all_desc1s = []
        bs = desc0.shape[0]

        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([ds_desc1, ds_desc0], dim=0), None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]

            elif name == 'self':  # self & cross share the pooled feature
                if self.pool_sizes is not None:
                    ds_desc0 = F.max_pool1d(desc0, self.pool_sizes[i])
                    ds_desc1 = F.max_pool1d(desc1, self.pool_sizes[i])
                else:
                    ds_desc0 = desc0
                    ds_desc1 = desc1

                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([ds_desc0, ds_desc1], dim=0), None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)
        return all_desc0s, all_desc1s, []


class DHGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, pool_sizes: list = None, minium_nghs: int = 128,
                 with_packing=False, ac_fn='relu'):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(num_heads=4, feature_dim=feature_dim, ac_fn=ac_fn)
            for i in range(len(layer_names))
        ])

        self.names = layer_names
        self.pool_sizes = pool_sizes
        self.minium_nghs = minium_nghs
        self.with_packing = with_packing

    def pooling_by_prob(self, X, global_indexes, prob, pool_size):
        '''
        :param X: [B, D, N]
        :param prob: [B, K, C] C <= N
        :param pool_size:
        :return:
        '''
        assert pool_size > 0
        if pool_size == 1:
            Y = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
            return Y, global_indexes

        sum_prob = torch.sum(prob, dim=1)  # [B, N]
        k = np.max([int(prob.shape[2] // pool_size), self.minium_nghs])
        k = np.min([k, prob.shape[2]])
        values, indexes = torch.topk(
            sum_prob,
            dim=1,
            k=k,
            largest=True)
        indexes = torch.sort(indexes, dim=1)[0]  # we need sorted indexes!!!
        sel_indexes = torch.gather(global_indexes, index=indexes, dim=1)
        Y = torch.gather(X, index=sel_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
        # print(X.shape, global_indexes.shape, prob.shape, pool_size, Y.shape, sel_indexes.shape)

        return Y, sel_indexes

    def pooling_by_prob_with_packing(self, X, global_indexes, prob, pool_size):
        '''
        :param X: [B, D, N]
        :param prob: [B, K, C] C <= N
        :param pool_size:
        :return:
        '''
        assert pool_size > 0
        if pool_size == 1:
            Y = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
            return Y, global_indexes, None

        if len(prob.shape) == 4:  # [B, H, K, N]
            sum_prob = torch.mean(prob, dim=1)
            sum_prob = torch.sum(sum_prob, dim=1)  # [B, N]
        else:
            sum_prob = torch.sum(prob, dim=1)  # [B, N]
        k = np.max([int(prob.shape[2] // pool_size), self.minium_nghs])
        k = np.min([k, prob.shape[2]])
        # values, indexes = torch.topk(
        #     sum_prob,
        #     dim=1,
        #     k=k,
        #     largest=True)
        # indexes = torch.sort(indexes, dim=1)[0]  # we need sorted indexes!!!
        # sel_indexes = torch.gather(global_indexes, index=indexes, dim=1)
        # Y = torch.gather(X, index=sel_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
        # print(X.shape, global_indexes.shape, prob.shape, pool_size, Y.shape, sel_indexes.shape)

        values, indexes = torch.sort(sum_prob, dim=1, descending=True)
        # print('idx: ', indexes.shape)
        # print('values: ', values.shape)
        info_indexes = indexes[:, :k]
        # info_values = values[:k]
        info_indexes = torch.sort(info_indexes, dim=1)[0]
        # print('values: ', values)
        # print('info: ', info_indexes)
        # print('global: ', global_indexes)
        sel_indexes = torch.gather(global_indexes, index=info_indexes, dim=1)
        Y = torch.gather(X, index=sel_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)

        if k == prob.shape[2] or not self.with_packing:  # no discarded tokens
            return Y, sel_indexes, None
        else:
            dis_indexes = indexes[:, k:]
            dis_values = values[:, k:]  # [B, N]
            dis_indexes = torch.gather(global_indexes, index=dis_indexes, dim=1)
            dis_Y = torch.gather(X, index=dis_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)  # [B, C, N]
            # print('disY: ', dis_Y.shape, dis_values.shape)
            packing_Y = self.packing(x=dis_Y, score=dis_values)

            return Y, sel_indexes, packing_Y

    def packing(self, x, score):
        '''
        :param x:
        :param score:
        :return:
        '''
        norm_score = F.softmax(score, dim=1)
        # print('norm: ', norm_score.shape)
        y = x * norm_score[:, None, :].expand_as(x)
        y = torch.sum(y, dim=2, keepdim=True)  # [B, C, 1]
        return y

    def forward(self, desc0, desc1):
        if self.training:
            return self.forward_train(desc0=desc0, desc1=desc1)
        else:
            # print('Im here')
            return self.forward_test(desc0=desc0, desc1=desc1)

    def forward_train(self, desc0, desc1):
        all_desc0s = []
        all_desc1s = []
        bs = desc0.shape[0]

        self_indexes = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs * 2, 1)
        cross_indexes = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs * 2, 1)
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                if self.pool_sizes is not None and self.pool_sizes[i] > 0:
                    ds_descs, cross_indexes, ds_packs = self.pooling_by_prob_with_packing(
                        X=torch.cat([desc0, desc1], dim=0),
                        global_indexes=cross_indexes,
                        prob=cross_prob,
                        pool_size=self.pool_sizes[i])
                    ds_desc0 = ds_descs[:bs]
                    ds_desc1 = ds_descs[bs:]
                    # print('ds_cross: ', i, name, cross_indexes[0:2, :8])
                else:
                    ds_desc0 = desc0
                    ds_desc1 = desc1
                    ds_packs = None

                if ds_packs is not None:
                    ds_desc0 = torch.cat([ds_desc0, ds_packs[:bs]], dim=2)
                    ds_desc1 = torch.cat([ds_desc1, ds_packs[bs:]], dim=2)
                    delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([ds_desc1, ds_desc0], dim=0), None)
                    cross_prob = torch.cat([layer.attn.prob[bs:], layer.attn.prob[:bs]], dim=0)  # [1->0, 0->1]
                    cross_prob = cross_prob[:, :, :-1]  # remove the pack
                else:
                    delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([ds_desc1, ds_desc0], dim=0), None)
                    cross_prob = torch.cat([layer.attn.prob[bs:], layer.attn.prob[:bs]], dim=0)  # [1->0, 0->1]
                delta0 = delta[:bs]
                delta1 = delta[bs:]

            elif name == 'self':  # self & cross share the pooled feature
                if self.pool_sizes is not None and self.pool_sizes[i] > 0:
                    ds_descs, self_indexes, ds_packs = self.pooling_by_prob_with_packing(
                        X=torch.cat([desc0, desc1], dim=0),
                        global_indexes=self_indexes,
                        prob=self_prob,
                        pool_size=self.pool_sizes[i])
                    # print('ds_self: ', ds_descs.shape, self_indexes.shape)
                    ds_desc0 = ds_descs[:bs]
                    ds_desc1 = ds_descs[bs:]
                    # print('ds_self: ', i, name, self_indexes[:2, :8])
                else:
                    ds_packs = None
                    ds_desc0 = desc0
                    ds_desc1 = desc1

                if ds_packs is not None:
                    ds_desc0 = torch.cat([ds_desc0, ds_packs[:bs]], dim=2)
                    ds_desc1 = torch.cat([ds_desc1, ds_packs[bs:]], dim=2)

                    delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([ds_desc0, ds_desc1], dim=0), None)
                    self_prob = layer.attn.prob

                    self_prob = self_prob[:, :, :-1]
                else:
                    delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([ds_desc0, ds_desc1], dim=0), None)
                    self_prob = layer.attn.prob
                delta0 = delta[:bs]
                delta1 = delta[bs:]

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)
        return all_desc0s, all_desc1s, []

    def forward_test(self, desc0, desc1):
        all_desc0s = []
        all_desc1s = []
        bs = desc0.shape[0]

        self_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(desc0.shape[0], 1)
        self_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(desc1.shape[0], 1)
        cross_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(desc0.shape[0], 1)
        cross_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(desc1.shape[0], 1)
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                if self.pool_sizes is not None and self.pool_sizes[i] > 0:
                    ds_desc0, cross_indexes0, ds_packs0 = self.pooling_by_prob_with_packing(X=desc0,
                                                                                            global_indexes=cross_indexes0,
                                                                                            prob=cross_prob0,
                                                                                            pool_size=self.pool_sizes[
                                                                                                i])
                    ds_desc1, cross_indexes1, ds_packs1 = self.pooling_by_prob_with_packing(X=desc1,
                                                                                            global_indexes=cross_indexes1,
                                                                                            prob=cross_prob1,
                                                                                            pool_size=self.pool_sizes[
                                                                                                i])
                else:
                    ds_desc0 = desc0
                    ds_desc1 = desc1

                    ds_packs0 = None
                    ds_packs1 = None

                if ds_packs1 is not None:
                    ds_desc1 = torch.cat([ds_desc1, ds_packs1], dim=2)
                    delta0 = layer(desc0, ds_desc1, None)
                    cross_prob1 = layer.attn.prob
                    cross_prob1 = cross_prob1[:, :, :-1]
                else:
                    delta0 = layer(desc0, ds_desc1, None)
                    cross_prob1 = layer.attn.prob

                if ds_packs0 is not None:
                    ds_desc0 = torch.cat([ds_desc0, ds_packs0], dim=2)
                    delta1 = layer(desc1, ds_desc0, None)
                    cross_prob0 = layer.attn.prob
                    cross_prob0 = cross_prob0[:, :, :-1]
                else:
                    delta1 = layer(desc1, ds_desc0, None)
                    cross_prob0 = layer.attn.prob

            elif name == 'self':  # self & cross share the pooled feature
                if self.pool_sizes is not None and self.pool_sizes[i] > 0:
                    ds_desc0, self_indexes0, ds_packs0 = self.pooling_by_prob_with_packing(X=desc0,
                                                                                           global_indexes=self_indexes0,
                                                                                           prob=self_prob0,
                                                                                           pool_size=self.pool_sizes[i])
                    ds_desc1, self_indexes1, ds_packs1 = self.pooling_by_prob_with_packing(X=desc1,
                                                                                           global_indexes=self_indexes1,
                                                                                           prob=self_prob1,
                                                                                           pool_size=self.pool_sizes[i])
                else:
                    ds_desc0 = desc0
                    ds_desc1 = desc1

                    ds_packs0 = None
                    ds_packs1 = None

                if ds_packs0 is not None:
                    ds_desc0 = torch.cat([ds_desc0, ds_packs0], dim=2)
                    delta0 = layer(desc0, ds_desc0, None)
                    self_prob0 = layer.attn.prob
                    self_prob0 = self_prob0[:, :, :-1]
                else:
                    delta0 = layer(desc0, ds_desc0, None)
                    self_prob0 = layer.attn.prob

                if ds_packs1 is not None:
                    ds_desc1 = torch.cat([ds_desc1, ds_packs1], dim=2)
                    delta1 = layer(desc1, ds_desc1, None)
                    self_prob1 = layer.attn.prob
                    self_prob1 = self_prob1[:, :, :-1]
                else:
                    delta1 = layer(desc1, ds_desc1, None)
                    self_prob1 = layer.attn.prob

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)
        return all_desc0s, all_desc1s, []


class SDHGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, pool_sizes: list = None, minium_nghs: int = 128,
                 sharing_layers: list = None,
                 with_packing=False, ac_fn='relu', norm_fn='bn'):
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
        self.pool_sizes = pool_sizes
        self.minium_nghs = minium_nghs
        self.with_packing = with_packing

    def pooling_by_prob(self, X, global_indexes, prob, pool_size):
        '''
        :param X: [B, D, N]
        :param prob: [B, K, C] C <= N
        :param pool_size:
        :return:
        '''
        assert pool_size > 0
        if pool_size == 1:
            Y = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
            return Y, global_indexes

        sum_prob = torch.sum(prob, dim=1)  # [B, N]
        k = np.max([int(prob.shape[2] // pool_size), self.minium_nghs])
        k = np.min([k, prob.shape[2]])
        values, indexes = torch.topk(
            sum_prob,
            dim=1,
            k=k,
            largest=True)
        indexes = torch.sort(indexes, dim=1)[0]  # we need sorted indexes!!!
        sel_indexes = torch.gather(global_indexes, index=indexes, dim=1)
        Y = torch.gather(X, index=sel_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
        # print(X.shape, global_indexes.shape, prob.shape, pool_size, Y.shape, sel_indexes.shape)

        return Y, sel_indexes

    def pooling_by_prob_with_packing(self, X, global_indexes, prob, pool_size):
        '''
        :param X: [B, D, N]
        :param prob: [B, K, C] C <= N
        :param pool_size:
        :return:
        '''
        assert pool_size > 0
        if pool_size == 1:
            Y = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
            return Y, global_indexes, None

        # print('X: ', X.shape, global_indexes.shape, prob.shape, pool_size)

        # sum_prob = torch.sum(prob, dim=1)  # [B, N]
        if len(prob.shape) == 4:  # [B, H, K, N]
            sum_prob = torch.mean(prob, dim=1)
            sum_prob = torch.sum(sum_prob, dim=1)  # [B, N]
        else:
            sum_prob = torch.sum(prob, dim=1)  # [B, N]

        k = np.max([int(prob.shape[2] // pool_size), self.minium_nghs])
        k = np.min([k, prob.shape[2]])
        # values, indexes = torch.topk(
        #     sum_prob,
        #     dim=1,
        #     k=k,
        #     largest=True)
        # indexes = torch.sort(indexes, dim=1)[0]  # we need sorted indexes!!!
        # sel_indexes = torch.gather(global_indexes, index=indexes, dim=1)
        # Y = torch.gather(X, index=sel_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
        # print(X.shape, global_indexes.shape, prob.shape, pool_size, Y.shape, sel_indexes.shape)

        values, indexes = torch.sort(sum_prob, dim=1, descending=True)
        # print('idx: ', indexes.shape)
        # print('values: ', values.shape)
        info_indexes = indexes[:, :k]
        # info_values = values[:k]
        info_indexes = torch.sort(info_indexes, dim=1)[0]
        # print('values: ', values)
        # print('info: ', info_indexes)
        # print('global: ', global_indexes)
        sel_indexes = torch.gather(global_indexes, index=info_indexes, dim=1)
        Y = torch.gather(X, index=sel_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)

        if k == prob.shape[2] or not self.with_packing:  # no discarded tokens
            return Y, sel_indexes, None
        else:
            dis_indexes = indexes[:, k:]
            dis_values = values[:, k:]  # [B, N]
            dis_indexes = torch.gather(global_indexes, index=dis_indexes, dim=1)
            dis_Y = torch.gather(X, index=dis_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)  # [B, C, N]
            # print('disY: ', dis_Y.shape, dis_values.shape)
            packing_Y = self.packing(x=dis_Y, score=dis_values)

            return Y, sel_indexes, packing_Y

    def pooling_by_prob_with_packing_v2(self, X, last_packing, global_indexes, prob, pool_size):
        '''
        :param X: [B, D, N]
        :param prob: [B, K, C] C <= N
        :param pool_size:
        :return:
        '''
        assert pool_size > 0
        if pool_size == 1:
            Y = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
            return Y, global_indexes, last_packing

        # print('X: ', X.shape, global_indexes.shape, prob.shape, pool_size)

        # sum_prob = torch.sum(prob, dim=1)  # [B, N]
        if len(prob.shape) == 4:  # [B, H, K, N]
            sum_prob = torch.mean(prob, dim=1)
            sum_prob = torch.sum(sum_prob, dim=1)  # [B, N]
        else:
            sum_prob = torch.sum(prob, dim=1)  # [B, N]

        nG = global_indexes.shape[1]
        nP = sum_prob.shape[1]
        # if packing is used, nP = nG + 1, else nP = nG
        k = np.max([int(nG // pool_size), self.minium_nghs])
        k = np.min([k, nG])
        # values, indexes = torch.topk(
        #     sum_prob,
        #     dim=1,
        #     k=k,
        #     largest=True)
        # indexes = torch.sort(indexes, dim=1)[0]  # we need sorted indexes!!!
        # sel_indexes = torch.gather(global_indexes, index=indexes, dim=1)
        # Y = torch.gather(X, index=sel_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
        # print(X.shape, global_indexes.shape, prob.shape, pool_size, Y.shape, sel_indexes.shape)

        values, indexes = torch.sort(sum_prob[:, :nG], dim=1, descending=True)
        # print('idx: ', indexes.shape)
        # print('values: ', values.shape)
        info_indexes = indexes[:, :k]
        # info_values = values[:k]
        info_indexes = torch.sort(info_indexes, dim=1)[0]
        # print('values: ', values)
        # print('info: ', info_indexes)
        # print('global: ', global_indexes)
        sel_indexes = torch.gather(global_indexes, index=info_indexes, dim=1)
        Y = torch.gather(X, index=sel_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)

        if k == nP or not self.with_packing:  # no discarded tokens
            return Y, sel_indexes, last_packing
        else:
            if nP == nG:  # no packing in last iteration
                dis_indexes = indexes[:, k:]
                dis_values = values[:, k:]  # [B, N]
                dis_indexes = torch.gather(global_indexes, index=dis_indexes, dim=1)
                dis_Y = torch.gather(X, index=dis_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)  # [B, C, N]
            else:
                dis_indexes = indexes[:, k:nG]
                dis_values = values[:, k:nG]  # [B, N]
                dis_indexes = torch.gather(global_indexes, index=dis_indexes, dim=1)
                dis_Y = torch.gather(X, index=dis_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)  # [B, C, N]

                # put last packing at the end
                dis_Y = torch.cat([dis_Y, last_packing], dim=-1)
                dis_values = torch.cat([dis_values, sum_prob[:, nG:]], dim=-1)

            # generate new packing
            # print('disY: ', dis_Y.shape, dis_values.shape)
            new_packing = self.packing(x=dis_Y, score=dis_values)

            return Y, sel_indexes, new_packing

    def packing(self, x, score):
        '''
        :param x:
        :param score:
        :return:
        '''
        # norm_score = F.softmax(score, dim=1)
        norm_score = score / torch.sum(score, dim=1, keepdim=True)
        # print('norm: ', norm_score.shape)
        y = x * norm_score[:, None, :].expand_as(x)
        y = torch.sum(y, dim=2, keepdim=True)  # [B, C, 1]
        return y

    def forward(self, desc0, desc1):
        if self.training:
            return self.forward_train(desc0=desc0, desc1=desc1)
        else:
            # print('Im here')
            return self.forward_test(desc0=desc0, desc1=desc1)

    def forward_train(self, desc0, desc1):
        all_desc0s = []
        all_desc1s = []

        # used for sharing attention
        self_attentions = [None]
        cross_attentions = [None]

        bs = desc0.shape[0]

        self_indexes = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs * 2, 1)
        cross_indexes = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs * 2, 1)

        last_cross_packs = None
        last_self_packs = None
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                if self.pool_sizes is not None and self.pool_sizes[i] > 0:
                    ds_descs, cross_indexes, ds_packs = self.pooling_by_prob_with_packing_v2(
                        X=torch.cat([desc0, desc1], dim=0),
                        last_packing=last_cross_packs,
                        global_indexes=cross_indexes,
                        prob=cross_prob,
                        pool_size=self.pool_sizes[i])
                    ds_desc0 = ds_descs[:bs]
                    ds_desc1 = ds_descs[bs:]
                    # print('ds_cross: ', i, name, cross_indexes[0:2, :8])
                else:
                    ds_desc0 = desc0
                    ds_desc1 = desc1
                    ds_packs = None

                if ds_packs is not None:
                    ds_desc0 = torch.cat([ds_desc0, ds_packs[:bs]], dim=2)
                    ds_desc1 = torch.cat([ds_desc1, ds_packs[bs:]], dim=2)

                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([ds_desc1, ds_desc0], dim=0),
                              cross_attentions[-1], None)
                cross_prob = torch.cat([layer.prob[bs:], layer.prob[:bs]], dim=0)  # [1->0, 0->1]
                cross_attentions.append(layer.prob)
                last_cross_packs = ds_packs

                delta0 = delta[:bs]
                delta1 = delta[bs:]

            elif name == 'self':  # self & cross share the pooled feature
                if self.pool_sizes is not None and self.pool_sizes[i] > 0:
                    ds_descs, self_indexes, ds_packs = self.pooling_by_prob_with_packing_v2(
                        X=torch.cat([desc0, desc1], dim=0),
                        last_packing=last_self_packs,
                        global_indexes=self_indexes,
                        prob=self_prob,
                        pool_size=self.pool_sizes[i])
                    # print('ds_self: ', ds_descs.shape, self_indexes.shape)
                    ds_desc0 = ds_descs[:bs]
                    ds_desc1 = ds_descs[bs:]
                    # print('ds_self: ', i, name, self_indexes[:2, :8])
                else:
                    ds_packs = None
                    ds_desc0 = desc0
                    ds_desc1 = desc1

                if ds_packs is not None:
                    ds_desc0 = torch.cat([ds_desc0, ds_packs[:bs]], dim=2)
                    ds_desc1 = torch.cat([ds_desc1, ds_packs[bs:]], dim=2)

                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([ds_desc0, ds_desc1], dim=0),
                              self_attentions[-1], None)
                self_prob = layer.prob
                self_attentions.append(layer.prob)

                last_self_packs = ds_packs
                delta0 = delta[:bs]
                delta1 = delta[bs:]

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)

        return all_desc0s, all_desc1s, []

    def forward_test(self, desc0, desc1):
        all_desc0s = []
        all_desc1s = []
        bs = desc0.shape[0]

        self_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(desc0.shape[0], 1)
        self_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(desc1.shape[0], 1)
        cross_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(desc0.shape[0], 1)
        cross_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(desc1.shape[0], 1)

        # used for sharing attention
        self_attentions00 = [None]
        self_attentions11 = [None]
        cross_attentions01 = [None]
        cross_attentions10 = [None]

        last_self_packs0 = None
        last_self_packs1 = None
        last_cross_packs0 = None
        last_cross_packs1 = None

        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                if self.pool_sizes is not None and self.pool_sizes[i] > 0:
                    ds_desc0, cross_indexes0, ds_packs0 = self.pooling_by_prob_with_packing_v2(
                        X=desc0,
                        last_packing=last_cross_packs0,
                        global_indexes=cross_indexes0,
                        prob=cross_prob0,
                        pool_size=self.pool_sizes[i])
                    ds_desc1, cross_indexes1, ds_packs1 = self.pooling_by_prob_with_packing_v2(
                        X=desc1,
                        last_packing=last_cross_packs1,
                        global_indexes=cross_indexes1,
                        prob=cross_prob1,
                        pool_size=self.pool_sizes[i])
                else:
                    ds_desc0 = desc0
                    ds_desc1 = desc1

                    ds_packs0 = last_cross_packs0
                    ds_packs1 = last_cross_packs0

                if ds_packs1 is not None:
                    ds_desc1 = torch.cat([ds_desc1, ds_packs1], dim=2)
                delta0 = layer(desc0, ds_desc1, cross_attentions01[-1], None)
                cross_prob1 = layer.prob
                cross_attentions01.append(layer.prob)

                if ds_packs0 is not None:
                    ds_desc0 = torch.cat([ds_desc0, ds_packs0], dim=2)
                delta1 = layer(desc1, ds_desc0, cross_attentions10[-1], None)
                cross_prob0 = layer.prob
                cross_attentions10.append(layer.prob)

                last_cross_packs0 = ds_packs0
                last_cross_packs1 = ds_packs1

            elif name == 'self':  # self & cross share the pooled feature
                if self.pool_sizes is not None and self.pool_sizes[i] > 0:
                    ds_desc0, self_indexes0, ds_packs0 = self.pooling_by_prob_with_packing_v2(
                        X=desc0,
                        last_packing=last_self_packs0,
                        global_indexes=self_indexes0,
                        prob=self_prob0,
                        pool_size=self.pool_sizes[i])
                    ds_desc1, self_indexes1, ds_packs1 = self.pooling_by_prob_with_packing_v2(
                        X=desc1,
                        last_packing=last_self_packs1,
                        global_indexes=self_indexes1,
                        prob=self_prob1,
                        pool_size=self.pool_sizes[i])
                else:
                    ds_desc0 = desc0
                    ds_desc1 = desc1

                    ds_packs0 = last_self_packs0
                    ds_packs1 = last_self_packs1

                if ds_packs0 is not None:
                    ds_desc0 = torch.cat([ds_desc0, ds_packs0], dim=2)
                delta0 = layer(desc0, ds_desc0, self_attentions00[-1], None)
                self_prob0 = layer.prob
                self_attentions00.append(layer.prob)

                if ds_packs1 is not None:
                    ds_desc1 = torch.cat([ds_desc1, ds_packs1], dim=2)
                delta1 = layer(desc1, ds_desc1, self_attentions11[-1], None)
                self_prob1 = layer.prob
                self_attentions11.append(layer.prob)

                last_self_packs0 = ds_packs0
                last_self_packs1 = ds_packs1

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)
        return all_desc0s, all_desc1s, []
