# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp-release -> eval_imp
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   24/03/2023 16:01
=================================================='''
import os.path as osp
import torch
import yaml
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch.utils.data as Data
from components.readers import standard_reader
from components.evaluators import auc_eval
from nets.superglue import SuperGlue
from nets.gm import GM
from nets.gms import DGNNS
from nets.adgm import AdaGMN
from eval.eval_yfcc_full import evaluate_full


def eval(model):
    thresholds = [5, 10, 20, 50]
    num_iterations = np.zeros(shape=(nI + 1, 1), dtype=int)


if __name__ == '__main__':
    feat = 'spp'
    dataset = 'yfcc'
    with_sinkhorn = True
    matching_method = 'SuperGlue'
    # matching_method = 'IMP'
    # matching_method = 'IMP_geo'
    # matching_method = 'EIMP'
    # matching_method = 'EIMP_geo'
    if dataset == 'scannet':
        if feat == 'spp':
            config_path = 'configs/scannet_eval_gm.yaml'
        else:
            config_path = 'configs/scannet_eval_gm_sift.yaml'
        error_th = 3
        # sh_its = 20
    elif dataset == 'yfcc':
        if feat == 'spp':
            config_path = 'configs/yfcc_eval_gm.yaml'
        else:
            config_path = 'configs/yfcc_eval_gm_sift.yaml'
        error_th = 1
        # sh_its = 20
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.Loader)
        read_config = config['reader']
        eval_config = config['evaluator']

    reader = standard_reader(config=read_config)
    reader_loader = Data.DataLoader(dataset=reader, num_workers=4, shuffle=False)
    evaluator = auc_eval(config=eval_config)

    config = {
        'descriptor_dim': 256 if feat == 'spp' else 128,
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,

        'with_sinkhorn': with_sinkhorn,
        'n_layers': 15,  # with sharing layers
        'GNN_layers': ['self', 'cross'] * 15,
        'ac_fn': 'relu',
        'norm_fn': 'in',
        'n_min_tokens': 256,
    }

    nI = 15

    if matching_method == 'IMP_geo':
        net = DGNNS(config=config)
        weight_path = '2022_09_09_19_20_39_dgnns_L15_megadepth_spp_B16_K1024_M0.2_relu_in_P512_MS_MP/dgnns.185.pth'  # scannet
        # weight_path = '2022_09_09_19_20_39_dgnns_L15_megadepth_spp_B16_K1024_M0.2_relu_in_P512_MS_MP/dgnns.190.pth' # yfcc
    elif matching_method == 'EIMP_geo':
        net = AdaGMN(config=config_ours)
        weight_path = '2022_10_06_19_55_55_adagmn_L15_megadepth_spp_B16_K1024_M0.2_relu_in_P512_MS_MP/adagmn.75.pth'  # scannet only

    weight_root = '/scratches/flyer_3/fx221/exp/pnba/'
    net.load_state_dict(torch.load(osp.join(weight_root, weight_path))['model'], strict=True)
    net = net.cuda().eval()

    reults = eval(model=net)
