# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def normalize_keypoints_from_HW(kpts, height, width):
    """ Normalize keypoints locations based on image image_shape"""
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
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
        # print('q1: ', query.shape, key.shape, value.shape)

        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)

        # print('q: ', query.shape, key.shape, value.shape, x.shape, prob.shape)
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


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)  # []b, m+1, n+1]

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,  # [self, cross, self, cross, ...] 9 in total
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
        'neg_ratio': 0.0,
        'require_prob': False,
        'layers': 9,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.config['GNN_layers'] = ['self', 'cross'] * self.config['layers']
        print('config: ', self.config)

        self.require_prob = self.config['require_prob']

        self.neg_ratio = self.config['neg_ratio']
        self.sinkhorn_iterations = self.config['sinkhorn_iterations']

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def produce_matches(self, data, p=0.2, **kwargs):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)

        if 'image0' in data.keys() and 'image1' in data.keys():
            # print('shape: ', data['image1'].shape)
            kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            kpts0 = data['norm_keypoints0']
            kpts1 = data['norm_keypoints1']

        # Keypoint MLP encoder.
        enc0 = self.kenc(kpts0, data['scores0'])  # [B, C, N]
        enc1 = self.kenc(kpts1, data['scores1'])
        # print('enc: ', enc0.shape, enc1.shape, desc0.shape, desc1.shape)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        if self.require_prob:
            desc0s, desc1s, prob = self.gnn(desc0, desc1, require_prob=True)
        else:
            desc0s, desc1s = self.gnn(desc0, desc1)
            prob = None

        nI = len(desc0s)
        nB = desc0.shape[0]
        desc0s = torch.vstack(desc0s)
        desc1s = torch.vstack(desc1s)
        mdescs0 = self.final_proj(desc0s)
        mdescs1 = self.final_proj(desc1s)
        # mdescs = self.final_proj(torch.vstack([desc0s, desc1s]))
        # dist = torch.einsum('bdn,bdm->bnm', mdescs[:nI * nB], mdescs[nI * nB:])
        dist = torch.einsum('bdn,bdm->bnm', mdescs0, mdescs1)
        dist = dist / self.config['descriptor_dim'] ** .5
        score = log_optimal_transport(dist, self.bin_score,
                                      iters=self.config['sinkhorn_iterations'])  # [nI * nB, N, M]
        # score = torch.exp(score)
        indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=score)

        # compute correct matches
        if 'matching_mask' in data.keys():
            gt_matching_mask = data['matching_mask'].repeat(nI, 1, 1)
            gt_matches = torch.max(gt_matching_mask[:, :-1, :], dim=-1, keepdim=False)[1]
            acc_corr = torch.sum(((indices0 - gt_matches) == 0) * (indices0 != -1) * (
                    gt_matches < gt_matching_mask.shape[-1] - 1)) / (nB * nI)
            acc_incorr = torch.sum((indices0 == -1) * (gt_matches == gt_matching_mask.shape[-1] - 1)) / (nB * nI)
            acc_corr_total = torch.sum((gt_matches < gt_matching_mask.shape[-1] - 1)) / (nB * nI)
            acc_incorr_total = torch.sum((gt_matches == gt_matching_mask.shape[-1] - 1)) / (nB * nI)
        else:
            acc_corr = torch.zeros(size=[], device=desc0.device) + 0
            acc_corr_total = torch.zeros(size=[], device=desc0.device) + 1
            acc_incorr = torch.zeros(size=[], device=desc0.device) + 0
            acc_incorr_total = torch.zeros(size=[], device=desc0.device) + 1

        if nI == 1:
            all_scores = [score]
            all_indices0 = [indices0]
            all_mscores0 = [mscores0]
        else:
            all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]
            all_indices0 = [indices0[i * nB: (i + 1) * nB] for i in range(nI)]
            all_mscores0 = [mscores0[i * nB: (i + 1) * nB] for i in range(nI)]

        return {
            'scores': all_scores,
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'acc_corr': [acc_corr],
            'acc_incorr': [acc_incorr],
            'total_acc_corr': [acc_corr_total],
            'total_acc_incorr': [acc_incorr_total],
        }

    def produce_matches_test(self, data, p=0.2, **kwargs):
        return self.produce_matches(data=data, p=p)

    def forward_train(self, data, p=0.2):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        # print(desc0.shape, desc1.shape, kpts0.shape, kpts1.shape, data['scores0'].shape, data['scores1'].shape)
        # print("norm: ", torch.sum(desc0 ** 2, dim=2)[0, 0:10])

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)
        # kpts0 = torch.reshape(kpts0, (1, -1, 2))
        # kpts1 = torch.reshape(kpts1, (1, -1, 2))

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }

        # Keypoint normalization.
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        enc0 = self.kenc(norm_kpts0, data['scores0'])  # [B, C, N]
        enc1 = self.kenc(norm_kpts1, data['scores1'])
        # print('enc: ', enc0.shape, enc1.shape, desc0.shape, desc1.shape)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        if self.require_prob:
            desc0s, desc1s, prob = self.gnn(desc0, desc1, require_prob=True)
        else:
            desc0s, desc1s = self.gnn(desc0, desc1)
            prob = None

        proj_desc0s = []
        proj_desc1s = []
        all_scores = []
        # print('desc0s: ', len(desc0s), len(desc1s))
        for mid, (desc0, desc1) in enumerate(zip(desc0s, desc1s)):
            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)  # [B, C, N]
            # print('mdesc: ', mdesc0.shape, mdesc1.shape)

            # Compute matching descriptor distance.
            score = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            score = score / self.config['descriptor_dim'] ** .5
            # Run the optimal transport.
            score = log_optimal_transport(
                score, self.bin_score,
                iters=self.config['sinkhorn_iterations'])
            all_scores.append(score)

        return proj_desc0s, proj_desc1s, all_scores, None

    def forward(self, data, mode=0):
        if not self.training:
            return self.produce_matches(data=data)
        else:
            return self.forward_train(data=data)

    def compute_matches(self, scores):
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return indices0, indices1, mscores0, mscores1


if __name__ == '__main__':
    config = {
        'superpoint': {
            'nms_radius': 3,
            'keypoint_threshold': 0.1,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': None,
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
            'descriptor_dim': 256,
            'neg_ratio': -1,
            'require_prob': False,
        }
    }

    nfeatures = 1024
    batch = 4
    M = 4
    kpts0 = torch.randint(0, 2, (batch, nfeatures, 2)).float().cuda()
    kpts1 = torch.randint(0, 2, (batch, nfeatures, 2)).float().cuda()
    scores0 = torch.rand((batch, nfeatures)).float().cuda()
    scores1 = torch.rand((batch, nfeatures)).float().cuda()
    descs0 = torch.rand((batch, nfeatures, 128)).float().cuda()
    descs1 = torch.rand((batch, nfeatures, 128)).float().cuda()
    descs0 = F.normalize(descs0, dim=1, p=2)
    descs1 = F.normalize(descs1, dim=1, p=2)

    data = {
        'keypoints0': kpts0,
        'keypoints1': kpts1,
        'scores0': scores0,
        'scores1': scores1,
        'descriptors0': descs0,
        'descriptors1': descs1,
        # 'labels0': labels0,
        # 'labels1': labels1,
        # 'confidences0': confidences0,
        # 'confidences1': confidences1,
        # 'matching_mask': matching_mask,
    }

    model = SuperGlue(config.get('superglue', {})).cuda().eval()
    for i in range(10):
        with torch.no_grad():
            out = model(data)
            prob = out['prob']
            print(prob.keys())
