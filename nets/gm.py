# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp-release -> gm
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   24/03/2023 10:49
=================================================='''
import torch
import torch.nn as nn

from nets.layers import MLP, KeypointEncoder, normalize_keypoints, sink_algorithm, arange_like, dual_softmax
from nets.loss import GraphLoss


class GM(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,  # [self, cross, self, cross, ...] 9 in total
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
        'with_pose': False,
        'n_layers': 9,
        'n_min_tokens': 256,
        'with_sinkhorn': True,

        'ac_fn': 'relu',
        'norm_fn': 'bn',
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        print('config in GM: ', self.config)

        self.neg_ratio = self.config['neg_ratio']
        self.multi_scale = self.config['multi_scale']
        self.multi_proj = self.config['multi_proj']
        self.n_layers = self.config['n_layers']

        self.with_sinkhorn = self.config['with_sinkhorn']
        self.match_threshold = self.config['match_threshold']

        self.sinkhorn_iterations = self.config['sinkhorn_iterations']
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'])
        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pooling_sizes=self.config['pooling_sizes'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'],
        )

        self.final_proj = nn.ModuleList([nn.Conv1d(
            self.config['descriptor_dim'],
            self.config['descriptor_dim'],
            kernel_size=1, bias=True) for _ in range(self.n_layers)])

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.match_net = GraphLoss(config=self.config)

        self.self_prob0 = None
        self.self_prob1 = None
        self.cross_prob0 = None
        self.cross_prob1 = None

    def forward_train(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)

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
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        desc0s, desc1s, all_matches = self.gnn(desc0, desc1)

        nI = len(desc0s)
        nB = desc0.shape[0]

        mdescs0 = []
        mdescs1 = []
        for l, d0, d1 in zip(self.final_proj, desc0s, desc1s):
            md = l(torch.vstack([d0, d1]))
            mdescs0.append(md[:nB])
            mdescs1.append(md[nB:])
        mdescs = torch.vstack([torch.vstack(mdescs0), torch.vstack(mdescs1)])

        dist = torch.einsum('bdn,bdm->bnm', mdescs[:nI * nB], mdescs[nI * nB:])
        dist = dist / self.config['descriptor_dim'] ** .5
        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

        loss_out = self.match_net(score, data['matching_mask'].repeat(nI, 1, 1))

        all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]
        loss_out['scores'] = all_scores
        loss = loss_out['matching_loss']

        loss_out['loss'] = loss

        return loss_out

    def produce_matches(self, data, p=0.2, only_last=False, **kwargs):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)

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
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        desc0s, desc1s, all_matches = self.gnn(desc0, desc1)

        nI = len(desc0s)
        nB = desc0.shape[0]
        if only_last:
            mdescs0 = self.final_proj[-1](desc0s[-1])
            mdescs1 = self.final_proj[-1](desc1s[-1])
        else:
            mdescs0 = []
            mdescs1 = []
            for l, d0, d1 in zip(self.final_proj, desc0s, desc1s):
                md0 = l(d0)
                md1 = l(d1)
                mdescs0.append(md0)
                mdescs1.append(md1)

            mdescs0 = torch.vstack(mdescs0)
            mdescs1 = torch.vstack(mdescs1)

        dist = torch.einsum('bdn,bdm->bnm', mdescs0, mdescs1)
        dist = dist / self.config['descriptor_dim'] ** .5
        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

        indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=score, p=p)

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
            if only_last:
                all_indices0 = [indices0]
                all_indices1 = [indices1]
                all_mscores0 = [mscores0]
                all_mscores1 = [mscores1]
                all_scores = [score]
            else:
                all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]
                all_indices0 = [indices0[i * nB: (i + 1) * nB] for i in range(nI)]
                all_mscores0 = [mscores0[i * nB: (i + 1) * nB] for i in range(nI)]

        output = {
            'scores': all_scores,
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'acc_corr': [acc_corr],
            'acc_incorr': [acc_incorr],
            'total_acc_corr': [acc_corr_total],
            'total_acc_incorr': [acc_incorr_total],
        }

        return output

    def produce_matches_test(self, data, p=0.2, only_last=False, **kwargs):
        return self.produce_matches(data=data, p=p, only_last=only_last, kwargs=kwargs)

    def forward(self, data, mode=0):
        if not self.training:
            if mode == 0:
                return self.produce_matches(data=data)
            else:
                return self.run(data=data)
        if self.with_pose:
            # return self.forward_train_with_pose(data=data)
            return self.forward_train_with_pose_v2(data=data)
        else:
            return self.forward_train(data=data)

    def forward_one_layer_old(self, desc0, desc1, M0, M1, layer_i):
        return self.gnn.forward_one_layer(desc0=desc0, desc1=desc1, M0=M0, M1=M1, layer_i=layer_i)

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        layer = self.gnn.layers[layer_i]
        name = self.gnn.names[layer_i]

        if name == 'cross':
            ds_desc0 = desc0
            ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc1, M=None)
            self.cross_prob1 = layer.prob
            delta1 = layer(desc1, ds_desc0, M=None)
            self.cross_prob0 = layer.prob

        elif name == 'self':
            ds_desc0 = desc0
            ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc0, M=None)
            self.self_prob0 = layer.prob
            delta1 = layer(desc1, ds_desc1, M=None)
            self.self_prob1 = layer.prob

        return desc0 + delta0, desc1 + delta1

    def encode_keypoint(self, norm_kpts0, norm_kpts1, scores0, scores1):
        return self.kenc(norm_kpts0, scores0), self.kenc(norm_kpts1, scores1)

    def compute_distance(self, desc0, desc1, layer_id=-1):
        if not self.multi_proj:
            mdesc0 = self.final_proj(desc0)
            mdesc1 = self.final_proj(desc1)
        else:
            mdesc0 = self.final_proj[layer_id](desc0)
            mdesc1 = self.final_proj[layer_id](desc1)
        dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        dist = dist / self.config['descriptor_dim'] ** .5
        return dist

    def compute_score(self, dist, dustbin, iteration):
        if self.with_sinkhorn:
            score = sink_algorithm(M=dist, dustbin=dustbin,
                                   iteration=iteration)  # [nI * nB, N, M]
        else:
            score = dual_softmax(M=dist, dustbin=dustbin)
        return score

    def compute_matches(self, scores, p=0.2):
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores0 = torch.where(mutual0, max0.values, zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        # valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid0 = mutual0 & (mscores0 > p)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return indices0, indices1, mscores0, mscores1

    def run(self, data):
        desc0 = data['desc1']
        # print('desc0: ', torch.sum(desc0 ** 2, dim=-1))
        # desc0 = torch.nn.functional.normalize(desc0, dim=-1)
        desc0 = desc0.transpose(1, 2)

        desc1 = data['desc2']
        # desc1 = torch.nn.functional.normalize(desc1, dim=-1)
        desc1 = desc1.transpose(1, 2)

        kpts0 = data['x1'][:, :, :2]
        kpts1 = data['x2'][:, :, :2]
        # kpts0 = normalize_keypoints(kpts=kpts0, image_shape=data['image_shape1'])
        # kpts1 = normalize_keypoints(kpts=kpts1, image_shape=data['image_shape2'])
        scores0 = data['x1'][:, :, -1]
        scores1 = data['x2'][:, :, -1]

        # Keypoint MLP encoder.
        enc0 = self.kenc(kpts0, scores0)  # [B, C, N]
        enc1 = self.kenc(kpts1, scores1)
        # print('enc: ', enc0.shape, enc1.shape, desc0.shape, desc1.shape)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        desc0s, desc1s, all_matches = self.gnn(desc0, desc1)

        desc0s = desc0s[-1]  # [nI * nB, C, N]
        desc1s = desc1s[-1]

        mdescs0 = self.final_proj[-1](desc0s)
        mdescs1 = self.final_proj[-1](desc1s)

        dist = torch.einsum('bdn,bdm->bnm', mdescs0, mdescs1)
        dist = dist / self.config['descriptor_dim'] ** .5
        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

        # print('score: ', score)
        output = {
            'p': score,
        }

        return output
