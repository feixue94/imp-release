# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp-release -> eval_imp
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   24/03/2023 16:01
=================================================='''
import torch
import yaml
import numpy as np
import cv2
import os.path as osp
import argparse
from tqdm import tqdm
import torch.utils.data as Data
from components.readers import standard_reader
from components.evaluators import auc_eval
from nets.gms import DGNNS
from nets.adgm import AdaGMN
from components.utils.evaluation_utils import normalize_intrinsic
from components.utils.metrics import compute_epi_inlier
from tools.utils import compute_pose_error, pose_auc
from tools.utils import estimate_pose_m_v2
from eval.matching import matching_iterative

parser = argparse.ArgumentParser(description='IMP', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--matching_method', type=str, default='IMP')
parser.add_argument('--dataset', type=str, default='scannet')
parser.add_argument('--feature_type', type=str, default='spp')
parser.add_argument('--use_dual_softmax', action='store_true', default=False)
parser.add_argument('--use_iterative', action='store_true', default=False)
parser.add_argument('--use_uncertainty', action='store_true', default=False)


def eval(model):
    thresholds = [5, 10, 20, 50]
    pose_errors = []
    precisions = []
    matching_scores = []
    num_iterations = np.zeros(shape=(nI + 1, 1), dtype=int)

    for index in tqdm(range(len(reader_loader)), total=len(reader_loader)):
        info = reader.run(index=index)
        img0, img1, E, R, t, K0, K1 = info['img1'], info['img2'], info['e'], info['r_gt'], info['t_gt'], info['K1'], \
                                      info['K2']
        x0, x1, descs0, descs1, size0, size1 = info['x1'], info['x2'], info['desc1'], info['desc2'], info['img1'].shape[
                                                                                                     :2], \
                                               info['img2'].shape[:2]

        pts0 = x0[:, :2]
        scores0 = x0[:, 2]
        pts1 = x1[:, :2]
        scores1 = x1[:, 2]

        norm_pts0 = normalize_intrinsic(x=pts0, K=K0)
        norm_pts1 = normalize_intrinsic(x=pts1, K=K1)

        T_0to1 = np.hstack([R, t.reshape(3, 1)])
        feed_data = {
            'keypoints0': torch.from_numpy(pts0).cuda().float()[None],
            'keypoints1': torch.from_numpy(pts1).cuda().float()[None],

            'image0': torch.from_numpy(img0).cuda().float()[None],
            'image1': torch.from_numpy(img1).cuda().float()[None],

            'descriptors0': torch.from_numpy(descs0).cuda().float()[None],
            'descriptors1': torch.from_numpy(descs1).cuda().float()[None],

            'scores0': torch.from_numpy(scores0).cuda().float()[None],
            'scores1': torch.from_numpy(scores1).cuda().float()[None],

            'K0': K0,
            'K1': K1,
            'T_0to1': T_0to1,
            'pts0_cpu': pts0,
            'pts1_cpu': pts1,
            'image_color0': img0,
            'image_color1': img1,
        }

        if 'rescale' in eval_config.keys():
            rescale = eval_config['rescale']
            size0, size1 = max(img0.shape), max(img1.shape)
            scale0, scale1 = rescale / size0, rescale / size1
        else:
            scale0, scale1 = 1.0, 1.0

        conf_th = 0.7
        stop_criteria = {
            'match': conf_th,
            'pose': 1.5,
        }

        if use_iterative:
            # matches, conf, pred_R, pred_t, ni = matching_iterative_v5(
            matches, conf, pred_R, pred_t, ni = matching_iterative(
                nI=nI,
                data=feed_data,
                model=model,
                error_th=error_th,
                stop_criteria=stop_criteria,
                match_ratio=0.1,
                method=cv2.USAC_MAGSAC,
                min_kpts=25,
            )

            valid = (matches > -1)
            mconf = conf[valid]
            pred_matches = np.vstack([np.where(matches > -1), matches[valid]]).transpose()
            # print(pred_matches.shape)

            mkpts0 = pts0[valid]
            mkpts1 = pts1[matches[valid]]

            # epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            # correct = epi_errs <= acc_error_th
            norm_mkpts0 = norm_pts0[valid]
            norm_mkpts1 = norm_pts1[matches[valid]]

            correct = compute_epi_inlier(x1=norm_mkpts0, x2=norm_mkpts1, E=E, inlier_th=0.005)

            num_correct = np.sum(correct)
            matching_score = num_correct / len(pts0) if len(pts0) > 0 else 0
            precision = np.mean(correct) if len(correct) > 0 else 0

            if pred_R is None:
                print('pred_R is None')
                ret = estimate_pose_m_v2(mkpts0, mkpts1, K0, K1, error_th, method=cv2.USAC_MAGSAC)
                if ret is None:
                    err_t, err_R = np.inf, np.inf
                else:
                    pred_R, pred_t, inliers = ret
                    if pred_R is not None:
                        R = pred_R
                        t = pred_t
                    err_t, err_R = compute_pose_error(T_0to1, R, t)
            else:
                err_t, err_R = compute_pose_error(T_0to1=T_0to1, R=pred_R, t=pred_t)

            for v in range(nI):
                if ni <= v + 1:
                    num_iterations[v + 1] += 1
        else:
            pred_R = None
            pred_t = None
            # match_out = net.produce_matches_test_R50(data={
            match_out = net.produce_matches(data={
                'keypoints0': torch.from_numpy(pts0).cuda().float()[None],
                'keypoints1': torch.from_numpy(pts1).cuda().float()[None],

                'image0': torch.from_numpy(img0).cuda().float().permute(2, 0, 1)[None],
                'image1': torch.from_numpy(img1).cuda().float().permute(2, 0, 1)[None],

                'descriptors0': torch.from_numpy(descs0).cuda().float()[None],
                'descriptors1': torch.from_numpy(descs1).cuda().float()[None],

                'scores0': torch.from_numpy(scores0).cuda().float()[None],
                'scores1': torch.from_numpy(scores1).cuda().float()[None],

                # 'matching_mask': torch.randint(0, 1, size=(2001, 2001)).cuda()[None]
            },
                p=0.2,
                only_last=True,
                # mscore_th=0.1,
                # uncertainty_ratio=1.,
            )
            indices0 = match_out['indices0']
            mscores0 = match_out['mscores0']
            # print(indices0.shape, mscores0.shape)
            print(len(indices0), len(mscores0))
            # exit(0)

            # if len(indices0.shape) == 4:  # [NI, B, N, M]
            if type(indices0) == list:  # [NI, B, N, M]
                matches = indices0[-1][0].cpu().numpy()
                conf = mscores0[-1][0].cpu().numpy()
            else:
                matches = indices0[0].cpu().numpy()
                conf = mscores0[0].cpu().numpy()

            num_iterations[nI] += 1

            valid = (matches > -1)
            mconf = conf[valid]

            pred_matches = np.vstack([np.where(matches > -1), matches[valid]]).transpose()
            # print(pred_matches.shape)

            mkpts0 = pts0[valid]
            mkpts1 = pts1[matches[valid]]
            # epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            # correct = epi_errs <= acc_error_th
            norm_mkpts0 = norm_pts0[valid]
            norm_mkpts1 = norm_pts1[matches[valid]]
            correct, epi_errs = compute_epi_inlier(x1=norm_mkpts0, x2=norm_mkpts1, E=E, inlier_th=0.005,
                                                   return_error=True)

            num_correct = np.sum(correct)
            matching_score = num_correct / len(pts0) if len(pts0) > 0 else 0
            precision = np.mean(correct) if len(correct) > 0 else 0

            ret = estimate_pose_m_v2(mkpts0, mkpts1, K0, K1, error_th, method=cv2.USAC_MAGSAC)
            # ret = estimate_pose_m(mkpts0, mkpts1, K0, K1, error_th, method=method)

            if ret is None:
                err_t, err_R = np.inf, np.inf
            else:
                R, t, inliers = ret
                if pred_R is not None:
                    R = pred_R
                    t = pred_t
                err_t, err_R = compute_pose_error(T_0to1, R, t)

        print('err_R: {:.2f}, err_t: {:.2f}'.format(err_R, err_t))

        pose_errors.append(np.max([err_R, err_t]))
        precisions.append(precision)
        matching_scores.append(matching_score)
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100. * yy for yy in aucs]
        prec = 100. * np.mean(precisions)
        ms = 100. * np.mean(matching_scores)
        print('Evaluation Results (mean over {} pairs):'.format(len(pose_errors)))
        print('AUC@5\t AUC@10\t AUC@20\t AUC@50\t Prec\t MScore\t Mkpts \t Ikpts')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], aucs[3], prec, ms))
        # print('Its: ')
        # print(num_iterations.shape, len(pose_errors), num_iterations[1])
        for ni in range(nI):
            print('It {:d} with {:.2f}'.format(ni + 1, num_iterations[ni + 1, 0] / len(pose_errors)))


if __name__ == '__main__':
    args = parser.parse_args()
    feat = args.feature_type
    dataset = args.dataset
    use_iterative = args.use_iterative
    use_sinkhorn = (not args.use_dual_softmax)
    use_uncertainty = args.use_uncertainty
    matching_method = args.matching_method
    if dataset == 'scannet':
        if feat == 'spp':
            config_path = 'configs/scannet_eval_gm.yaml'
        else:
            config_path = 'configs/scannet_eval_gm_sift.yaml'
        error_th = 3
    elif dataset == 'yfcc':
        if feat == 'spp':
            config_path = 'configs/yfcc_eval_gm.yaml'
        else:
            config_path = 'configs/yfcc_eval_gm_sift.yaml'
        error_th = 1
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

        'with_sinkhorn': use_sinkhorn,
        'n_layers': 15,  # with sharing layers
        'GNN_layers': ['self', 'cross'] * 15,
        'ac_fn': 'relu',
        'norm_fn': 'in',
        'n_min_tokens': 256,
    }

    nI = 15

    model_dict = {
        'IMP_geo': {
            'network': DGNNS(config=config),
            'weight': {
                'scannet': '2022_09_09_19_20_39_dgnns_L15_megadepth_spp_B16_K1024_M0.2_relu_in_P512_MS_MP/dgnns.185.pth',
                'yfcc': '2022_09_09_19_20_39_dgnns_L15_megadepth_spp_B16_K1024_M0.2_relu_in_P512_MS_MP/dgnns.190.pth',
            }
        },
        'IMP': {
            'network': DGNNS(config=config),
            'weight': {
                'scannet': '2022_07_15_13_49_43_dgnns_L15_megadepth_spp_B16_K1024_M0.2_relu_in_MS_MP/dgnns.885.pth',
                'yfcc': '2022_07_15_13_49_43_dgnns_L15_megadepth_spp_B16_K1024_M0.2_relu_in_MS_MP/dgnns.885.pth',
            }
        },
        'EIMP': {
            'network': AdaGMN(config=config),
            'weight': {
                'scannet': '2022_10_06_15_06_23_adagmn_L15_megadepth_spp_B16_K1024_M0.2_relu_in_MS_MP/adagmn.100.pth',
                'yfcc': '2022_10_06_15_06_23_adagmn_L15_megadepth_spp_B16_K1024_M0.2_relu_in_MS_MP/adagmn.100.pth',
            }
        },
        'EIMP_geo': {
            'network': AdaGMN(config=config),
            'weight': {
                'scannet': '2022_10_06_19_55_55_adagmn_L15_megadepth_spp_B16_K1024_M0.2_relu_in_P512_MS_MP/adagmn.75.pth',
                'yfcc': '2022_10_06_19_55_55_adagmn_L15_megadepth_spp_B16_K1024_M0.2_relu_in_P512_MS_MP/adagmn.45.pth',
            }
        }
    }
    net = model_dict[matching_method]['network']
    weight_path = model_dict[matching_method]['weight'][dataset]
    weight_root = '/scratches/flyer_3/fx221/exp/pnba/'
    net.load_state_dict(state_dict=torch.load(osp.join(weight_root, weight_path))['model'], strict=True)
    net = net.cuda().eval()

    with torch.no_grad():
        reults = eval(model=net)

    print(
        'Results of model {} on {} dataset (iterative {}, sinkhorn {}, uncertainty {}'.format(matching_method,
                                                                                              dataset,
                                                                                              use_iterative,
                                                                                              use_sinkhorn,
                                                                                              use_uncertainty))
