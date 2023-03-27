# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp-release -> matching
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   26/03/2023 14:56
=================================================='''
import torch
import numpy as np
import cv2
from nets.gm import normalize_keypoints
from tools.utils import compute_pose_error, angle_error_mat, angle_error_vec
from eval.pose_estimation import estimate_pose


def matching_iterative(data, model, nI, match_ratio, min_kpts, error_th, stop_criteria, method=cv2.USAC_MAGSAC):
    pts0 = data['keypoints0']
    pts1 = data['keypoints1']

    if 'norm_keypoint0' in data.keys() and 'norm_keypoint1' in data.keys():
        norm_kpts0 = data['norm_keypoints0']
        norm_kpts1 = data['norm_keypoints1']
    else:
        norm_kpts0 = normalize_keypoints(kpts=pts0, image_shape=data['image0'].shape)
        norm_kpts1 = normalize_keypoints(kpts=pts1, image_shape=data['image1'].shape)
    scores0 = data['scores0']
    scores1 = data['scores1']
    desc0 = data['descriptors0']
    desc1 = data['descriptors1']
    desc0 = desc0.transpose(1, 2)
    desc1 = desc1.transpose(1, 2)

    pts0_cpu = data['pts0_cpu']
    pts1_cpu = data['pts1_cpu']

    K0 = data['K0']
    K1 = data['K1']
    T_0to1 = data['T_0to1']

    last_best_R = None
    last_best_t = None

    valid_its = [3, 5, 7, 9, 11, 13, 14]

    for it in range(nI):
        if it == 0:
            enc0, enc1 = model.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                               scores1=scores1)
            desc0 = desc0 + enc0
            desc1 = desc1 + enc1
        desc0, desc1 = model.forward_one_layer(desc0=desc0, desc1=desc1, M0=None, M1=None, layer_i=it * 2)
        desc0, desc1 = model.forward_one_layer(desc0=desc0, desc1=desc1, M0=None, M1=None, layer_i=it * 2 + 1)

        # only perform pose estimation at specified layers
        if it not in valid_its:
            continue

        pred_dist = model.compute_distance(desc0=desc0, desc1=desc1, layer_id=it)
        pred_score = model.compute_score(dist=pred_dist, dustbin=model.bin_score,
                                         iteration=model.sinkhorn_iterations)
        indices0, indices1, mscores0, mscores1 = model.compute_matches(scores=pred_score, p=match_ratio)

        if torch.sum(indices0 > -1) < min_kpts:
            last_best_R = None
            last_best_t = None
            continue

        indices0_cpu = indices0[0].cpu().numpy()
        mscores0_cpu = mscores0[0].cpu().numpy()
        pred_score_cpu = pred_score.cpu().numpy()[0, :-1, :-1]

        matched_ids0 = [v for v in range(indices0_cpu.shape[0]) if
                        indices0_cpu[v] > -1]
        matched_ids1 = [indices0_cpu[v] for v in range(indices0_cpu.shape[0]) if
                        indices0_cpu[v] > -1]
        matched_score = [mscores0_cpu[v] for v in matched_ids0]

        pred_matches = np.vstack([matched_ids0, matched_ids1]).transpose()
        mscore = np.array(matched_score)

        if pred_matches.shape[0] == 0:
            continue

        ret = estimate_pose(kpts0=pts0_cpu[pred_matches[:, 0]],
                            kpts1=pts1_cpu[pred_matches[:, 1]],
                            K0=K0, K1=K1,
                            norm_thresh=error_th, method=method)

        if ret is not None:
            E, R, t, inliers = ret
            error_t, error_R = compute_pose_error(T_0to1=T_0to1, R=R, t=t)
            pose_inliers = inliers
        else:
            R, t = None, None
            error_R, error_t = np.inf, np.inf
            pose_inliers = np.array([False for v in range(pred_matches.shape[0])])
        if it >= 1:
            diff_R = angle_error_mat(R1=last_best_R,
                                     R2=R) if last_best_R is not None and R is not None else np.inf
            diff_t = angle_error_vec(v1=last_best_t,
                                     v2=t) if last_best_t is not None and t is not None else np.inf
        else:
            diff_R, diff_t = np.inf, np.inf

        pose_diff = np.max([diff_R, diff_t])
        last_best_R = R
        last_best_t = t

        # Check if stop iteration
        if 'pose' in stop_criteria.keys():
            if pose_diff <= stop_criteria['pose']:
                output_indice0 = np.zeros_like(indices0_cpu) - 1
                output_indice0[pred_matches[pose_inliers, 0]] = pred_matches[pose_inliers, 1]

                print('diff: ', diff_R, diff_t, np.sum(pose_inliers), pred_matches.shape[0])

                return output_indice0, mscores0_cpu, R, t, it + 1

    indices0, indices1, mscores0, mscores1 = model.compute_matches(scores=pred_score, p=0.2)
    indices0_cpu = indices0[0].cpu().numpy()
    mscores0_cpu = mscores0[0].cpu().numpy()

    return indices0_cpu, mscores0_cpu, None, None, nI


def matching_iterative_uncertainty(data, model, nI, match_ratio, min_kpts, error_th, stop_criteria,
                                   method=cv2.USAC_MAGSAC, with_uncertainty=False):
    pts0 = data['keypoints0']
    pts1 = data['keypoints1']

    if 'norm_keypoint0' in data.keys() and 'norm_keypoint1' in data.keys():
        norm_kpts0 = data['norm_keypoints0']
        norm_kpts1 = data['norm_keypoints1']
    else:
        norm_kpts0 = normalize_keypoints(kpts=pts0, image_shape=data['image0'].shape)
        norm_kpts1 = normalize_keypoints(kpts=pts1, image_shape=data['image1'].shape)
    scores0 = data['scores0']
    scores1 = data['scores1']
    desc0 = data['descriptors0']
    desc1 = data['descriptors1']
    desc0 = desc0.transpose(1, 2)
    desc1 = desc1.transpose(1, 2)

    pts0_cpu = data['pts0_cpu']
    pts1_cpu = data['pts1_cpu']

    K0 = data['K0']
    K1 = data['K1']
    T_0to1 = data['T_0to1']

    last_best_R = None
    last_best_t = None

    valid_its = [3, 5, 7, 9, 11, 13, 14]

    sel_ids0 = None
    sel_ids1 = None
    enc0, enc1 = model.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0, scores1=scores1)
    desc0 = desc0 + enc0
    desc1 = desc1 + enc1

    update_0 = False
    update_1 = False

    for it in range(nI):
        if update_0:
            desc0 = desc0[:, :, sel_ids0]
            pts0_cpu = pts0_cpu[sel_ids0.cpu().numpy()]
            norm_kpts0 = norm_kpts0[:, sel_ids0, :]

        if update_1:
            desc1 = desc1[:, :, sel_ids1]
            pts1_cpu = pts1_cpu[sel_ids1.cpu().numpy()]
            norm_kpts1 = norm_kpts1[:, sel_ids1.cpu(), :]

        desc0, desc1 = model.forward_one_layer(desc0=desc0, desc1=desc1, M0=None, M1=None, layer_i=it * 2)
        desc0, desc1 = model.forward_one_layer(desc0=desc0, desc1=desc1, M0=None, M1=None, layer_i=it * 2 + 1)

        # only perform pose estimation at certain layers
        if it not in valid_its:
            update_0 = False
            update_1 = False
            continue

        prob00 = model.self_prob0
        prob11 = model.self_prob1
        prob01 = model.cross_prob0
        prob10 = model.cross_prob1

        pred_dist = model.compute_distance(desc0=desc0, desc1=desc1, layer_id=it)
        pred_score = model.compute_score(dist=pred_dist, dustbin=model.bin_score,
                                         iteration=model.sinkhorn_iterations)
        indices0, indices1, mscores0, mscores1 = model.compute_matches(scores=pred_score, p=match_ratio)

        if torch.sum(indices0 > -1) < min_kpts:
            last_best_R = None
            last_best_t = None
            continue

        indices0_cpu = indices0[0].cpu().numpy()
        mscores0_cpu = mscores0[0].cpu().numpy()
        pred_score_cpu = pred_score.cpu().numpy()[0, :-1, :-1]

        matched_ids0 = [v for v in range(indices0_cpu.shape[0]) if
                        indices0_cpu[v] > -1]
        matched_ids1 = [indices0_cpu[v] for v in range(indices0_cpu.shape[0]) if
                        indices0_cpu[v] > -1]
        matched_score = [mscores0_cpu[v] for v in matched_ids0]

        pred_matches = np.vstack([matched_ids0, matched_ids1]).transpose()
        mscore = np.array(matched_score)

        if pred_matches.shape[0] == 0:
            continue

        ret = estimate_pose(kpts0=pts0_cpu[pred_matches[:, 0]],
                            kpts1=pts1_cpu[pred_matches[:, 1]],
                            K0=K0, K1=K1,
                            norm_thresh=error_th, method=method)

        if ret is not None:
            E, R, t, inliers = ret
            error_t, error_R = compute_pose_error(T_0to1=T_0to1, R=R, t=t)
            pose_inliers = inliers
            inlier_ratio = np.sum(pose_inliers) / pred_matches.shape[0]
        else:
            E, R, t = None, None, None
            error_R, error_t = np.inf, np.inf
            pose_inliers = np.array([False for v in range(pred_matches.shape[0])])
            inlier_ratio = 0
        if it >= 1:
            diff_R = angle_error_mat(R1=last_best_R,
                                     R2=R) if last_best_R is not None and R is not None else np.inf
            diff_t = angle_error_vec(v1=last_best_t,
                                     v2=t) if last_best_t is not None and t is not None else np.inf
        else:
            diff_R, diff_t = np.inf, np.inf

        pose_diff = np.max([diff_R, diff_t])
        last_best_R = R
        last_best_t = t

        # performing adaptive pooling
        if with_uncertainty:
            if inlier_ratio == 0:
                mscore_th = 0.2
            else:
                mscore_th = 0.2 * inlier_ratio

            print('inlier ratio, mscore_th: ', inlier_ratio, mscore_th)
        else:
            mscore_th = 0.2

        sel_ids0, sel_ids1 = model.pool(pred_score=pred_score, prob00=prob00, prob01=prob01, prob11=prob11,
                                        prob10=prob10, mscore_th=mscore_th, uncertainty_ratio=1.0)
        update_0 = False if sel_ids0 is None else True
        update_1 = False if sel_ids1 is None else True

        # Check if stop iteration
        if 'pose' in stop_criteria.keys():
            if pose_diff <= stop_criteria['pose']:
                output_indice0 = np.zeros_like(indices0_cpu) - 1
                output_indice0[pred_matches[pose_inliers, 0]] = pred_matches[pose_inliers, 1]

                print('diff: ', diff_R, diff_t, np.sum(pose_inliers), pred_matches.shape[0])

                return pts0_cpu, pts1_cpu, norm_kpts0[0].cpu().numpy(), norm_kpts1[
                    0].cpu().numpy(), output_indice0, mscores0_cpu, R, t, it + 1
                # return output_indice0, mscores0_cpu, R, t, it + 1

    indices0, indices1, mscores0, mscores1 = model.compute_matches(scores=pred_score, p=0.2)
    indices0_cpu = indices0[0].cpu().numpy()
    mscores0_cpu = mscores0[0].cpu().numpy()

    return pts0_cpu, pts1_cpu, norm_kpts0[0].cpu().numpy(), \
           norm_kpts1[0].cpu().numpy(), indices0_cpu, mscores0_cpu, None, None, nI
