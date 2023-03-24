from .transformations import quaternion_from_matrix
import numpy as np
import os
import sys


def evaluate_R_t(R_gt, t_gt, R, t):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt) ** 2))
    err_t = np.arccos(np.sqrt(1 - loss_t))
    return np.rad2deg(err_q), np.rad2deg(err_t)


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds[1:]:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


def approx_pose_auc(errors, thresholds):
    qt_acc_hist, _ = np.histogram(errors, thresholds)
    num_pair = float(len(errors))
    qt_acc_hist = qt_acc_hist.astype(float) / num_pair
    qt_acc = np.cumsum(qt_acc_hist)
    approx_aucs = [np.mean(qt_acc[:i]) for i in range(1, len(thresholds))]
    return approx_aucs


def compute_epi_inlier(x1, x2, E, inlier_th, return_error=False):
    num_pts1, num_pts2 = x1.shape[0], x2.shape[0]
    x1_h = np.concatenate([x1, np.ones([num_pts1, 1])], -1)
    x2_h = np.concatenate([x2, np.ones([num_pts2, 1])], -1)
    ep_line1 = x1_h @ E.T
    ep_line2 = x2_h @ E
    norm_factor = (1 / np.sqrt((ep_line1[:, :2] ** 2).sum(1)) + 1 / np.sqrt((ep_line2[:, :2] ** 2).sum(1))) / 2
    dis = abs((ep_line1 * x2_h).sum(-1)) * norm_factor
    inlier_mask = dis < inlier_th

    if return_error:
        return inlier_mask, dis
    else:
        return inlier_mask
