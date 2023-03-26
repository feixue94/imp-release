# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp-release -> gms
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   24/03/2023 11:45
=================================================='''
import torch
import torch.nn as nn
from nets.gm import GM
from nets.layers import SAGNN
from nets.layers import normalize_keypoints


class DGNNS(GM):
    def __init__(self, config={}):
        sharing_layers = [False, False] * 2 + [False, False, True, True] * 21
        super().__init__(config=config)

        self.gnn = SAGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'],
            sharing_layers=sharing_layers,
        )

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

            if only_last:
                if ni == nI - 1:
                    mdesc0 = self.final_proj[ni](desc0)
                    mdesc1 = self.final_proj[ni](desc1)

                    dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                    dist = dist / self.config['descriptor_dim'] ** .5
                    pred_score = self.compute_score(dist=dist, dustbin=self.bin_score,
                                                    iteration=self.sinkhorn_iterations)

                    indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
                    all_indices0.append(indices0)
                    all_mscores0.append(mscores0)
            else:
                mdesc0 = self.final_proj[ni](desc0)
                mdesc1 = self.final_proj[ni](desc1)

                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score,
                                                iteration=self.sinkhorn_iterations)

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

    def run(self, data):
        # used for evaluation
        desc0 = data['desc1']
        desc1 = data['desc2']
        kpts0 = data['x1'][:, :, :2]
        kpts1 = data['x2'][:, :, :2]
        scores0 = data['x1'][:, :, -1]
        scores1 = data['x2'][:, :, -1]

        out = self.produce_matches(
            data={
                'descriptors0': desc0,
                'descriptors1': desc1,
                'norm_keypoints0': kpts0,
                'norm_keypoints1': kpts1,
                'scores0': scores0,
                'scores1': scores1,
            },
            p=self.config['match_threshold'],
            only_last=True,
        )
        # out = self.produce_matches(data=data)
        indices0 = out['indices0'][-1][0]
        # indices1 = out['indices0'][-1][0]
        index0 = torch.where(indices0 >= 0)[0]
        index1 = indices0[index0]

        return {
            'index0': index0,
            'index1': index1,
        }

    def pool(self, **kwargs):
        return None, None
