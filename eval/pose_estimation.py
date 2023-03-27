# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp-release -> pose_estimation
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   26/03/2023 16:32
=================================================='''
from copy import deepcopy
import numpy as np
import cv2


def decompose_essential_mat(E, pts0, pts1, K0, K1, distance_thresh=1000):
    def get_mask_from_pts4D(pts4D, P):
        Q = deepcopy(pts4D)
        mask = (Q[2] * Q[3]) > 0
        Q[0] /= Q[3]
        Q[1] /= Q[3]
        Q[2] /= Q[3]
        Q[3] /= Q[3]

        mask = (Q[2] < distance_thresh) * mask
        Q = P @ Q
        mask = (Q[2] > 0) * mask
        mask = (Q[2] < distance_thresh) * mask

        return mask

    K = (K0 + K1) / 2.
    pts0[:, 0] = (pts0[:, 0] - K[0, 2]) / K[0, 0]
    pts0[:, 1] = (pts0[:, 1] - K[1, 2]) / K[1, 1]
    pts1[:, 0] = (pts1[:, 0] - K[0, 2]) / K[0, 0]
    pts1[:, 1] = (pts1[:, 1] - K[1, 2]) / K[1, 1]

    pts0 = pts0.transpose()
    pts1 = pts1.transpose()

    R1, R2, t = cv2.decomposeEssentialMat(E=E)
    t = t.reshape(3, )
    P0 = np.eye(3, 4)
    P1 = np.zeros(shape=(3, 4), dtype=R1.dtype)
    P2 = np.zeros(shape=(3, 4), dtype=R1.dtype)
    P3 = np.zeros(shape=(3, 4), dtype=R1.dtype)
    P4 = np.zeros(shape=(3, 4), dtype=R1.dtype)

    P1[0:3, 0:3] = R1
    P1[0:3, 3] = t

    P2[0:3, 0:3] = R2
    P2[0:3, 3] = t

    P3[0:3, 0:3] = R1
    P3[0:3, 3] = -t

    P4[0:3, 0:3] = R2
    P4[0:3, 3] = -t

    pts4d1 = cv2.triangulatePoints(P0, P1, pts0, pts1)
    mask1 = get_mask_from_pts4D(pts4D=pts4d1, P=P1)

    pts4d2 = cv2.triangulatePoints(P0, P2, pts0, pts1)
    mask2 = get_mask_from_pts4D(pts4D=pts4d2, P=P2)

    pts4d3 = cv2.triangulatePoints(P0, P3, pts0, pts1)
    mask3 = get_mask_from_pts4D(pts4D=pts4d3, P=P3)

    pts4d4 = cv2.triangulatePoints(P0, P4, pts0, pts1)
    mask4 = get_mask_from_pts4D(pts4D=pts4d4, P=P4)

    good1 = np.sum(mask1)
    good2 = np.sum(mask2)
    good3 = np.sum(mask3)
    good4 = np.sum(mask4)

    # print('P1: ', R1, t, good1)
    # print('P2: ', R2, t, good2)
    # print('P3: ', R1, -t, good3)
    # print('P4: ', R2, -t, good4)

    best_good = np.max([good1, good2, good3, good4])

    if good1 == best_good:
        return R1, t, mask1
    elif good2 == best_good:
        return R2, t, mask2
    elif good3 == best_good:
        return R1, -t, mask3
    else:
        return R2, -t, mask4


def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999, method=cv2.RANSAC, mask=None):
    if len(kpts0) < 5:
        return None

    E, E_mask = cv2.findEssentialMat(points1=kpts0,
                                     points2=kpts1,
                                     cameraMatrix1=K0,
                                     cameraMatrix2=K1,
                                     distCoeffs1=None,
                                     distCoeffs2=None,
                                     threshold=norm_thresh,
                                     prob=conf,
                                     mask=mask,
                                     method=method)

    if E is None or E.shape[0] != 3 or E.shape[1] != 3:
        return None

    R, t, mask_P = decompose_essential_mat(E=E, pts0=kpts0[E_mask.ravel() > 0], pts1=kpts1[E_mask.ravel() > 0],
                                           K0=K0, K1=K1)

    mask = E_mask.ravel() >= 0
    mask[E_mask.ravel() > 0] = mask_P
    return R, t, mask
