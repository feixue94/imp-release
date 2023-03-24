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
