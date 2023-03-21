# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> gmn
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   16/05/2022 09:39
=================================================='''
import torch
import torch.nn as nn
from nets.gm import *
from nets.layers import *


class GMN(GM):
    def __init__(self, config={}):
        super().__init__(config=config)
        self.gnn = AttentionalGNN(
            # k=self.config['n_cluster'],
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers']
        )

        self.n_connection_self = self.config['n_connection_self']
        self.n_connection_cross = self.config['n_connection_cross']


class HGNN(GM):
    def __init__(self, config={}):
        pool_sizes = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 64, 64, 128, 128, 256,
                      256]
        super().__init__(config={**config, **{'pool_sizes': pool_sizes}})
        self.gnn = HAGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pool_sizes=pool_sizes,
        )

        self.n_connection_self = self.config['n_connection_self']
        self.n_connection_cross = self.config['n_connection_cross']


# class DGNN(GM):
#     def __init__(self, config={}):
#         pool_sizes = [0, 0] * 2 + [2, 2, 1, 1] * 15
#         super().__init__(config={**config, **{'pool_sizes': pool_sizes}})
#         self.gnn = DHGNN(
#             feature_dim=self.config['descriptor_dim'],
#             layer_names=self.config['GNN_layers'],
#             pool_sizes=pool_sizes,
#             with_packing=False,
#             ac_fn=self.config['ac_fn'],
#         )
#
#         self.pool_sizes = pool_sizes
#         self.self_indexes0 = None
#         self.self_indexes1 = None
#         self.self_prob0 = None
#         self.self_prob1 = None
#
#         self.cross_indexes0 = None
#         self.cross_indexes1 = None
#         self.cross_prob0 = None
#         self.cross_prob1 = None
#
#     def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
#         bs = desc0.shape[0]
#         layer = self.gnn.layers[layer_i]
#         name = self.gnn.names[layer_i]
#         ps = self.pool_sizes[layer_i]
#
#         if layer_i == 0:
#             self.self_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs, 1)
#             self.cross_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs, 1)
#
#             self.self_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(bs, 1)
#             self.cross_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(bs, 1)
#
#         if name == 'cross':
#             if ps > 0:
#                 ds_desc0, cross_indexes0 = self.gnn.pooling_by_prob(X=desc0, global_indexes=self.cross_indexes0,
#                                                                     prob=self.cross_prob0, pool_size=ps)
#                 ds_desc1, cross_indexes1 = self.gnn.pooling_by_prob(X=desc1, global_indexes=self.cross_indexes1,
#                                                                     prob=self.cross_prob1, pool_size=ps)
#
#                 self.cross_indexes0 = cross_indexes0
#                 self.cross_indexes1 = cross_indexes1
#             else:
#                 ds_desc0 = desc0
#                 ds_desc1 = desc1
#
#             delta0 = layer(desc0, ds_desc1, None)
#             self.cross_prob1 = layer.attn.prob
#
#             delta1 = layer(desc1, ds_desc0, None)
#             self.cross_prob0 = layer.attn.prob
#         else:
#             if ps > 0:
#                 ds_desc0, self_indexes0 = self.gnn.pooling_by_prob(X=desc0, global_indexes=self.self_indexes0,
#                                                                    prob=self.self_prob0, pool_size=ps)
#                 ds_desc1, self_indexes1 = self.gnn.pooling_by_prob(X=desc1, global_indexes=self.self_indexes1,
#                                                                    prob=self.self_prob1, pool_size=ps)
#                 self.self_indexes0 = self_indexes0
#                 self.self_indexes1 = self_indexes1
#             else:
#                 ds_desc0 = desc0
#                 ds_desc1 = desc1
#
#             delta0 = layer(desc0, ds_desc0, None)
#             self.self_prob0 = layer.attn.prob
#
#             delta1 = layer(desc1, ds_desc1, None)
#             self.self_prob1 = layer.attn.prob
#
#         # print(name, torch.numel(self.self_indexes0), torch.numel(self.cross_indexes0),
#         #       torch.numel(self.self_indexes1), torch.numel(self.cross_indexes1))
#
#         return desc0 + delta0, desc1 + delta1


# class DGNNP(GM):
#     def __init__(self, config={}):
#         pool_sizes = [0, 0] * 2 + [2, 2, 1, 1] * 15
#         super().__init__(config={**config, **{'pool_sizes': pool_sizes}})
#         self.gnn = DHGNN(
#             feature_dim=self.config['descriptor_dim'],
#             layer_names=self.config['GNN_layers'],
#             pool_sizes=pool_sizes,
#             with_packing=True,
#             ac_fn=self.config['ac_fn'],
#         )
#
#         self.pool_sizes = pool_sizes
#         self.self_indexes0 = None
#         self.self_indexes1 = None
#         self.self_prob0 = None
#         self.self_prob1 = None
#
#         self.cross_indexes0 = None
#         self.cross_indexes1 = None
#         self.cross_prob0 = None
#         self.cross_prob1 = None
#
#     def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
#         bs = desc0.shape[0]
#         layer = self.gnn.layers[layer_i]
#         name = self.gnn.names[layer_i]
#         ps = self.pool_sizes[layer_i]
#
#         if layer_i == 0:
#             self.self_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs, 1)
#             self.cross_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs, 1)
#
#             self.self_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(bs, 1)
#             self.cross_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(bs, 1)
#
#         if name == 'cross':
#             if ps > 0:
#                 ds_desc0, cross_indexes0, ds_packs0 = self.gnn.pooling_by_prob_with_packing(X=desc0,
#                                                                                             global_indexes=self.cross_indexes0,
#                                                                                             prob=self.cross_prob0,
#                                                                                             pool_size=ps)
#                 ds_desc1, cross_indexes1, ds_packs1 = self.gnn.pooling_by_prob_with_packing(X=desc1,
#                                                                                             global_indexes=self.cross_indexes1,
#                                                                                             prob=self.cross_prob1,
#                                                                                             pool_size=ps)
#
#                 self.cross_indexes0 = cross_indexes0
#                 self.cross_indexes1 = cross_indexes1
#             else:
#                 ds_desc0 = desc0
#                 ds_desc1 = desc1
#
#                 ds_packs0 = None
#                 ds_packs1 = None
#
#             if ds_packs1 is not None:
#                 ds_desc1 = torch.cat([ds_desc1, ds_packs1], dim=2)
#                 delta0 = layer(desc0, ds_desc1, None)
#                 self.cross_prob1 = layer.prob
#                 self.cross_prob1 = self.cross_prob1[:, :, :-1]  # remove pack
#             else:
#                 delta0 = layer(desc0, ds_desc1, None)
#                 self.cross_prob1 = layer.prob
#
#             if ds_packs0 is not None:
#                 ds_desc0 = torch.cat([ds_desc0, ds_packs0], dim=2)
#                 delta1 = layer(desc1, ds_desc0, None)
#                 self.cross_prob0 = layer.prob
#                 self.cross_prob0 = self.cross_prob0[:, :, :-1]
#             else:
#                 delta1 = layer(desc1, ds_desc0, None)
#                 self.cross_prob0 = layer.prob
#         else:
#             if ps > 0:
#                 ds_desc0, self_indexes0, ds_packs0 = self.gnn.pooling_by_prob_with_packing(X=desc0,
#                                                                                            global_indexes=self.self_indexes0,
#                                                                                            prob=self.self_prob0,
#                                                                                            pool_size=ps)
#                 ds_desc1, self_indexes1, ds_packs1 = self.gnn.pooling_by_prob_with_packing(X=desc1,
#                                                                                            global_indexes=self.self_indexes1,
#                                                                                            prob=self.self_prob1,
#                                                                                            pool_size=ps)
#                 self.self_indexes0 = self_indexes0
#                 self.self_indexes1 = self_indexes1
#             else:
#                 ds_desc0 = desc0
#                 ds_desc1 = desc1
#                 ds_packs0 = None
#                 ds_packs1 = None
#
#             if ds_packs0 is not None:
#                 ds_desc0 = torch.cat([ds_desc0, ds_packs0], dim=2)
#                 delta0 = layer(desc0, ds_desc0, None)
#                 self.self_prob0 = layer.prob
#                 self.self_prob0 = self.self_prob0[:, :, :-1]
#             else:
#                 delta0 = layer(desc0, ds_desc0, None)
#                 self.self_prob0 = layer.prob
#
#             if ds_packs1 is not None:
#                 ds_desc1 = torch.cat([ds_desc1, ds_packs1], dim=2)
#                 delta1 = layer(desc1, ds_desc1, None)
#                 self.self_prob1 = layer.prob
#                 # print('sp1-1: ', self.self_prob1.shape)
#                 self.self_prob1 = self.self_prob1[:, :, :-1]
#                 # print('sp1-2: ', self.self_prob1.shape)
#
#             else:
#                 delta1 = layer(desc1, ds_desc1, None)
#                 self.self_prob1 = layer.prob
#
#         # print(name, torch.numel(self.self_indexes0), torch.numel(self.cross_indexes0),
#         #       torch.numel(self.self_indexes1), torch.numel(self.cross_indexes1))
#
#         return desc0 + delta0, desc1 + delta1


class DGNNPS(GM):
    def __init__(self, config={}):
        pool_sizes = [0, 0] * 2 + [2, 2, 1, 1] * 21
        # sharing_layers = [False, False] * 2 + [False, False, True, True] * 21
        sharing_layers = [False, False] * 2 + [False, False, True, True] * 21
        super().__init__(config={**config, **{'pool_sizes': pool_sizes}})
        self.gnn = SDHGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pool_sizes=pool_sizes,
            with_packing=True,
            ac_fn=self.config['ac_fn'],
            sharing_layers=sharing_layers,
            norm_fn=self.config['norm_fn'],
        )

        self.pool_sizes = pool_sizes
        self.self_indexes0 = None
        self.self_indexes1 = None
        self.self_prob0 = None
        self.self_prob1 = None

        self.cross_indexes0 = None
        self.cross_indexes1 = None
        self.cross_prob0 = None
        self.cross_prob1 = None

        # self.self_attns00 = [None]
        # self.self_attns11 = [None]
        # self.cross_attns01 = [None]
        # self.cross_attns10 = [None]
        self.self_attn00 = None
        self.self_attn11 = None
        self.cross_attn01 = None
        self.cross_attn10 = None

        self.last_self_packs0 = None
        self.last_self_packs1 = None
        self.last_cross_packs0 = None
        self.last_cross_packs1 = None

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        bs = desc0.shape[0]
        layer = self.gnn.layers[layer_i]
        name = self.gnn.names[layer_i]
        ps = self.pool_sizes[layer_i]

        if layer_i == 0:
            self.self_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs, 1)
            self.cross_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs, 1)

            self.self_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(bs, 1)
            self.cross_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(bs, 1)

        if name == 'cross':
            if ps > 0:
                ds_desc0, cross_indexes0, ds_packs0 = self.gnn.pooling_by_prob_with_packing_v2(
                    X=desc0,
                    last_packing=self.last_cross_packs0,
                    global_indexes=self.cross_indexes0,
                    prob=self.cross_prob0,
                    pool_size=ps)
                ds_desc1, cross_indexes1, ds_packs1 = self.gnn.pooling_by_prob_with_packing_v2(
                    X=desc1,
                    last_packing=self.last_cross_packs1,
                    global_indexes=self.cross_indexes1,
                    prob=self.cross_prob1,
                    pool_size=ps)

                self.cross_indexes0 = cross_indexes0
                self.cross_indexes1 = cross_indexes1
            else:
                ds_desc0 = desc0
                ds_desc1 = desc1

                ds_packs0 = self.last_cross_packs0
                ds_packs1 = self.last_cross_packs1

            if ds_packs1 is not None:
                ds_desc1 = torch.cat([ds_desc1, ds_packs1], dim=2)
            delta0 = layer(desc0, ds_desc1, self.cross_attn01, None)
            self.cross_prob1 = layer.prob
            self.cross_attn01 = layer.prob

            if ds_packs0 is not None:
                ds_desc0 = torch.cat([ds_desc0, ds_packs0], dim=2)
            delta1 = layer(desc1, ds_desc0, self.cross_attn10, None)
            self.cross_prob0 = layer.prob
            self.cross_attn10 = layer.prob
        else:
            if ps > 0:
                ds_desc0, self_indexes0, ds_packs0 = self.gnn.pooling_by_prob_with_packing_v2(
                    X=desc0,
                    last_packing=self.last_self_packs0,
                    global_indexes=self.self_indexes0,
                    prob=self.self_prob0,
                    pool_size=ps)
                ds_desc1, self_indexes1, ds_packs1 = self.gnn.pooling_by_prob_with_packing_v2(
                    X=desc1,
                    last_packing=self.last_self_packs1,
                    global_indexes=self.self_indexes1,
                    prob=self.self_prob1,
                    pool_size=ps)
                self.self_indexes0 = self_indexes0
                self.self_indexes1 = self_indexes1
            else:
                ds_desc0 = desc0
                ds_desc1 = desc1
                ds_packs0 = self.last_self_packs0
                ds_packs1 = self.last_self_packs1

            if ds_packs0 is not None:
                ds_desc0 = torch.cat([ds_desc0, ds_packs0], dim=2)
            delta0 = layer(desc0, ds_desc0, self.self_attn00, None)
            self.self_prob0 = layer.prob
            self.self_attn00 = layer.prob

            if ds_packs1 is not None:
                ds_desc1 = torch.cat([ds_desc1, ds_packs1], dim=2)
            delta1 = layer(desc1, ds_desc1, self.self_attn11, None)
            self.self_prob1 = layer.prob
            self.self_attn11 = layer.prob

        return desc0 + delta0, desc1 + delta1


class DGNNP(DGNNPS):
    def __init__(self, config={}):
        pool_sizes = [0, 0] * 2 + [2, 2, 1, 1] * 21
        # sharing_layers = [False, False] * 2 + [False, False, False, False] * 21
        sharing_layers = [False, False] * 2 + [False, False, True, True] * 21
        super().__init__(config={**config, **{'pool_sizes': pool_sizes}})

        self.gnn = SDHGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pool_sizes=pool_sizes,
            with_packing=True,
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'],
            sharing_layers=sharing_layers,
        )

        self.pool_sizes = pool_sizes
        self.self_indexes0 = None
        self.self_indexes1 = None
        self.self_prob0 = None
        self.self_prob1 = None

        self.cross_indexes0 = None
        self.cross_indexes1 = None
        self.cross_prob0 = None
        self.cross_prob1 = None

        self.self_attns00 = [None]
        self.self_attns11 = [None]
        self.cross_attns01 = [None]
        self.cross_attns10 = [None]

        self.last_self_packs0 = None
        self.last_self_packs1 = None
        self.last_cross_packs0 = None
        self.last_cross_packs1 = None


class DGNNS(DGNNPS):
    def __init__(self, config={}):
        pool_sizes = [0, 0] * 2 + [0, 0, 0, 0] * 21
        sharing_layers = [False, False] * 2 + [False, False, True, True] * 21
        super().__init__(config={**config, **{'pool_sizes': pool_sizes}})

        self.gnn = SDHGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pool_sizes=pool_sizes,
            with_packing=True,
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'],
            sharing_layers=sharing_layers,
        )

        self.pool_sizes = pool_sizes
        self.self_indexes0 = None
        self.self_indexes1 = None
        self.self_prob0 = None
        self.self_prob1 = None

        self.cross_indexes0 = None
        self.cross_indexes1 = None
        self.cross_prob0 = None
        self.cross_prob1 = None

        self.self_attns00 = [None]
        self.self_attns11 = [None]
        self.cross_attns01 = [None]
        self.cross_attns10 = [None]

        self.last_self_packs0 = None
        self.last_self_packs1 = None
        self.last_cross_packs0 = None
        self.last_cross_packs1 = None

    def produce_matches_vis(self, data, p=0.2, test_it=-1, only_last=False, **kwargs):
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

        all_indices0 = []
        all_mscores0 = []

        all_prob00 = []
        all_prob01 = []
        all_prob11 = []
        all_prob10 = []

        prob00 = None
        prob11 = None
        prob01 = None
        prob10 = None

        nI = self.config['n_layers']

        if test_it >= 1 and test_it <= nI:
            nI = test_it

        for ni in range(nI):
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0,
                           prob=prob00,
                           M=None)
            prob00 = layer.prob

            delta1 = layer(desc1, desc1,
                           prob=prob11,
                           M=None)
            prob11 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1,
                           prob=prob10,
                           M=None)
            prob10 = layer.prob

            delta1 = layer(desc1, desc0,
                           prob=prob01,
                           M=None)
            prob01 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            all_prob00.append(prob00)
            all_prob11.append(prob11)
            all_prob10.append(prob10)
            all_prob01.append(prob01)

            if not self.multi_proj:
                mdesc0 = self.final_proj(desc0)
                mdesc1 = self.final_proj(desc1)
            else:
                mdesc0 = self.final_proj[ni](desc0)
                mdesc1 = self.final_proj[ni](desc1)

            dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            dist = dist / self.config['descriptor_dim'] ** .5
            # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
            pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

            indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
            all_indices0.append(indices0)
            all_mscores0.append(mscores0)

        out = {
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'prob00': all_prob00,
            'prob01': all_prob01,
            'prob11': all_prob11,
            'prob10': all_prob10,
        }
        return out

    def test_time(self, data, p=0.2, test_it=-1, **kwargs):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

        # desc0 = desc0.transpose(1, 2)  # [B, D, N]
        # desc1 = desc1.transpose(1, 2)

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

        prob00 = None
        prob11 = None
        prob01 = None
        prob10 = None
        nI = self.config['n_layers']

        if test_it >= 1 and test_it <= nI:
            nI = test_it

        for ni in range(nI):
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0, prob=prob00, M=None)
            prob00 = layer.prob

            delta1 = layer(desc1, desc1, prob=prob11, M=None)
            prob11 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1, prob=prob10, M=None)
            prob10 = layer.prob

            delta1 = layer(desc1, desc0, prob=prob01, M=None)
            prob01 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            if ni == nI - 1:
                if not self.multi_proj:
                    mdesc0 = self.final_proj(desc0)
                    mdesc1 = self.final_proj(desc1)
                else:
                    mdesc0 = self.final_proj[ni](desc0)
                    mdesc1 = self.final_proj[ni](desc1)

                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                # indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)

        out = {
            'pred_score': pred_score,
        }
        return out

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        layer = self.gnn.layers[layer_i]
        name = self.gnn.names[layer_i]

        if name == 'cross':
            ds_desc0 = desc0
            ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc1, self.cross_prob1, M=None)
            self.cross_prob1 = layer.prob
            delta1 = layer(desc1, ds_desc0, self.cross_prob0, M=None)
            self.cross_prob0 = layer.prob

        elif name == 'self':
            ds_desc0 = desc0
            ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc0, self.self_prob0, M=None)
            self.self_prob0 = layer.prob
            delta1 = layer(desc1, ds_desc1, self.self_prob1, M=None)
            self.self_prob1 = layer.prob

        return desc0 + delta0, desc1 + delta1

    def pool(self, **kwargs):
        return None, None


class DGNN(DGNNPS):
    def __init__(self, config={}):
        pool_sizes = [0, 0] * 2 + [0, 0, 0, 0] * 21
        sharing_layers = [False, False] * 2 + [False, False, False, False] * 21
        super().__init__(config={**config, **{'pool_sizes': pool_sizes}})

        self.gnn = SDHGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pool_sizes=pool_sizes,
            with_packing=True,
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'],
            sharing_layers=sharing_layers,
        )

        self.pool_sizes = pool_sizes
        self.self_indexes0 = None
        self.self_indexes1 = None
        self.self_prob0 = None
        self.self_prob1 = None

        self.cross_indexes0 = None
        self.cross_indexes1 = None
        self.cross_prob0 = None
        self.cross_prob1 = None

        self.self_attns00 = [None]
        self.self_attns11 = [None]
        self.cross_attns01 = [None]
        self.cross_attns10 = [None]

        self.last_self_packs0 = None
        self.last_self_packs1 = None
        self.last_cross_packs0 = None
        self.last_cross_packs1 = None

    def produce_matches_vis(self, data, p=0.2, test_it=-1, only_last=False, **kwargs):
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

        all_indices0 = []
        all_mscores0 = []

        all_prob00 = []
        all_prob01 = []
        all_prob11 = []
        all_prob10 = []

        prob00 = None
        prob11 = None
        prob01 = None
        prob10 = None

        nI = self.config['n_layers']

        if test_it >= 1 and test_it <= nI:
            nI = test_it

        for ni in range(nI):
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0,
                           prob=prob00,
                           M=None)
            prob00 = layer.prob

            delta1 = layer(desc1, desc1,
                           prob=prob11,
                           M=None)
            prob11 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1,
                           prob=prob10,
                           M=None)
            prob10 = layer.prob

            delta1 = layer(desc1, desc0,
                           prob=prob01,
                           M=None)
            prob01 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            all_prob00.append(prob00)
            all_prob11.append(prob11)
            all_prob10.append(prob10)
            all_prob01.append(prob01)

            if not self.multi_proj:
                mdesc0 = self.final_proj(desc0)
                mdesc1 = self.final_proj(desc1)
            else:
                mdesc0 = self.final_proj[ni](desc0)
                mdesc1 = self.final_proj[ni](desc1)

            dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            dist = dist / self.config['descriptor_dim'] ** .5
            # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
            pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

            indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
            all_indices0.append(indices0)
            all_mscores0.append(mscores0)

        out = {
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'prob00': all_prob00,
            'prob01': all_prob01,
            'prob11': all_prob11,
            'prob10': all_prob10,
        }
        return out

    def test_time(self, data, p=0.2, test_it=-1, **kwargs):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

        # desc0 = desc0.transpose(1, 2)  # [B, D, N]
        # desc1 = desc1.transpose(1, 2)

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

        prob00 = None
        prob11 = None
        prob01 = None
        prob10 = None
        nI = self.config['n_layers']

        if test_it >= 1 and test_it <= nI:
            nI = test_it

        for ni in range(nI):
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0, prob=prob00, M=None)
            prob00 = layer.prob

            delta1 = layer(desc1, desc1, prob=prob11, M=None)
            prob11 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1, prob=prob10, M=None)
            prob10 = layer.prob

            delta1 = layer(desc1, desc0, prob=prob01, M=None)
            prob01 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            if ni == nI - 1:
                if not self.multi_proj:
                    mdesc0 = self.final_proj(desc0)
                    mdesc1 = self.final_proj(desc1)
                else:
                    mdesc0 = self.final_proj[ni](desc0)
                    mdesc1 = self.final_proj[ni](desc1)

                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                # indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)

        out = {
            'pred_score': pred_score,
        }
        return out

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        layer = self.gnn.layers[layer_i]
        name = self.gnn.names[layer_i]

        if name == 'cross':
            ds_desc0 = desc0
            ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc1, self.cross_prob1, M=None)
            self.cross_prob1 = layer.prob
            delta1 = layer(desc1, ds_desc0, self.cross_prob0, M=None)
            self.cross_prob0 = layer.prob

        elif name == 'self':
            ds_desc0 = desc0
            ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc0, self.self_prob0, M=None)
            self.self_prob0 = layer.prob
            delta1 = layer(desc1, ds_desc1, self.self_prob1, M=None)
            self.self_prob1 = layer.prob

        return desc0 + delta0, desc1 + delta1

    def pool(self, **kwargs):
        return None, None


class DGNNV2(GM):
    def __init__(self, config={}):
        pool_sizes = [0, 0, 2, 2] + [1, 1, 2, 2] * 15
        super().__init__(config={**config, **{'pool_sizes': pool_sizes}})
        self.gnn = DHGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pool_sizes=pool_sizes,
        )

        self.pool_sizes = pool_sizes
        self.self_indexes0 = None
        self.self_indexes1 = None
        self.self_prob0 = None
        self.self_prob1 = None

        self.cross_indexes0 = None
        self.cross_indexes1 = None
        self.cross_prob0 = None
        self.cross_prob1 = None

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        bs = desc0.shape[0]
        layer = self.gnn.layers[layer_i]
        name = self.gnn.names[layer_i]
        ps = self.pool_sizes[layer_i]

        if layer_i == 0:
            self.self_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs, 1)
            self.cross_indexes0 = torch.arange(0, desc0.shape[2], device=desc0.device)[None].repeat(bs, 1)

            self.self_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(bs, 1)
            self.cross_indexes1 = torch.arange(0, desc1.shape[2], device=desc0.device)[None].repeat(bs, 1)

        if name == 'cross':
            if ps > 0:
                ds_desc0, cross_indexes0 = self.gnn.pooling_by_prob(X=desc0, global_indexes=self.cross_indexes0,
                                                                    prob=self.cross_prob0, pool_size=ps)
                ds_desc1, cross_indexes1 = self.gnn.pooling_by_prob(X=desc1, global_indexes=self.cross_indexes1,
                                                                    prob=self.cross_prob1, pool_size=ps)

                self.cross_indexes0 = cross_indexes0
                self.cross_indexes1 = cross_indexes1
            else:
                ds_desc0 = desc0
                ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc1, None)
            self.cross_prob1 = layer.attn.prob

            delta1 = layer(desc1, ds_desc0, None)
            self.cross_prob0 = layer.attn.prob

        else:
            if ps > 0:
                ds_desc0, self_indexes0 = self.gnn.pooling_by_prob(X=desc0, global_indexes=self.self_indexes0,
                                                                   prob=self.self_prob0, pool_size=ps)
                ds_desc1, self_indexes1 = self.gnn.pooling_by_prob(X=desc1, global_indexes=self.self_indexes1,
                                                                   prob=self.self_prob1, pool_size=ps)

                self.self_indexes0 = self_indexes0
                self.self_indexes1 = self_indexes1
            else:
                ds_desc0 = desc0
                ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc0, None)
            self.self_prob0 = layer.attn.prob

            delta1 = layer(desc1, ds_desc1, None)
            self.self_prob1 = layer.attn.prob

        return desc0 + delta0, desc1 + delta1
