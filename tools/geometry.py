# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> geometry
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   10/12/2021 14:43
=================================================='''
import numpy as np
import cv2
import torch
from tools.utils import to_homogeneous
import math


def reproject(pos1, depth1, intrinsics1, pose1, bbox1, intrinsics2, pose2, bbox2):
    Z1 = depth1[pos1.astype(int)[1, :], pos1.astype(int)[0, :]]

    # COLMAP convention
    if bbox1 is not None:
        u1 = pos1[0, :] + bbox1[1] + .5
        v1 = pos1[1, :] + bbox1[0] + .5
    else:
        u1 = pos1[0, :] + .5
        v1 = pos1[1, :] + .5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    XYZ1_hom = np.vstack([
        X1.reshape(1, -1),
        Y1.reshape(1, -1),
        Z1.reshape(1, -1),
        np.ones_like(Z1).reshape(1, -1),
    ])

    XYZ2_hom = (pose2 @ np.linalg.inv(pose1)) @ XYZ1_hom
    XYZ2 = XYZ2_hom[:-1, :] / (XYZ2_hom[-1, :].reshape(1, -1) + 1e-5)
    uv2_hom = intrinsics2 @ XYZ2
    uv2 = uv2_hom[:-1, :] / (uv2_hom[-1, :].reshape(1, -1) + 1e-5)

    if bbox2 is not None:
        u2 = uv2[0, :] - bbox2[1] - .5
        v2 = uv2[1, :] - bbox2[0] - .5
    else:
        u2 = uv2[0, :] - .5
        v2 = uv2[1, :] - .5
    uv2 = np.vstack([u2.reshape(1, -1), v2.reshape(1, -1)])

    return uv2


def reproject_points(pos1, depth1, intrinsics1, pose1, bbox1, intrinsics2, pose2, bbox2):
    Z1 = depth1

    # COLMAP convention
    if bbox1 is not None:
        u1 = pos1[0, :] + bbox1[1] + .5
        v1 = pos1[1, :] + bbox1[0] + .5
    else:
        u1 = pos1[0, :] + .5
        v1 = pos1[1, :] + .5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    XYZ1_hom = np.vstack([
        X1.reshape(1, -1),
        Y1.reshape(1, -1),
        Z1.reshape(1, -1),
        np.ones_like(Z1).reshape(1, -1),
    ])

    XYZ2_hom = (pose2 @ np.linalg.inv(pose1)) @ XYZ1_hom
    XYZ2 = XYZ2_hom[:-1, :] / (XYZ2_hom[-1, :].reshape(1, -1) + 1e-5)
    uv2_hom = intrinsics2 @ XYZ2
    uv2 = uv2_hom[:-1, :] / (uv2_hom[-1, :].reshape(1, -1) + 1e-5)

    if bbox2 is not None:
        u2 = uv2[0, :] - bbox2[1] - .5
        v2 = uv2[1, :] - bbox2[0] - .5
    else:
        u2 = uv2[0, :] - .5
        v2 = uv2[1, :] - .5
    uv2 = np.vstack([u2.reshape(1, -1), v2.reshape(1, -1)])

    return uv2


def reproject_points_torch(pos1, depth1, intrinsics1, pose1, bbox1, intrinsics2, pose2, bbox2):
    Z1 = depth1

    # COLMAP convention
    if bbox1 is not None:
        u1 = pos1[0, :] + bbox1[1] + .5
        v1 = pos1[1, :] + bbox1[0] + .5
    else:
        u1 = pos1[0, :] + .5
        v1 = pos1[1, :] + .5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    XYZ1_hom = torch.vstack([
        X1.reshape(1, -1),
        Y1.reshape(1, -1),
        Z1.reshape(1, -1),
        torch.ones_like(Z1).reshape(1, -1),
    ])

    XYZ2_hom = (pose2.float() @ torch.linalg.inv(pose1).float()) @ XYZ1_hom.float()
    XYZ2 = XYZ2_hom[:-1, :] / (XYZ2_hom[-1, :].reshape(1, -1) + 1e-5)
    uv2_hom = intrinsics2 @ XYZ2
    uv2 = uv2_hom[:-1, :] / (uv2_hom[-1, :].reshape(1, -1) + 1e-5)

    if bbox2 is not None:
        u2 = uv2[0, :] - bbox2[1] - .5
        v2 = uv2[1, :] - bbox2[0] - .5
    else:
        u2 = uv2[0, :] - .5
        v2 = uv2[1, :] - .5
    uv2 = torch.vstack([u2.reshape(1, -1), v2.reshape(1, -1)])

    return uv2


def compute_reprojection_error(
        pos1, depth1, intrinsics1, pose1, bbox1,
        pos2, depth2, intrinsics2, pose2, bbox2):
    reproject_12 = reproject(pos1=pos1, depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=bbox1,
                             pose2=pose2, intrinsics2=intrinsics2, bbox2=bbox2)

    error = (reproject_12 - pos2) ** 2
    error = np.sqrt(np.sum(error, axis=0))
    return error


def match_from_projection(
        pos1, depth1, intrinsics1, pose1, bbox1,  # [2, N]
        pos2, depth2, intrinsics2, pose2, bbox2,
        inlier_th=3, outlier_th=5,
        cycle_check=False):  # [2, M]

    proj_uv12 = reproject(pos1=pos1, depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=bbox1,
                          intrinsics2=intrinsics2, pose2=pose2, bbox2=bbox2)  # [2, N]
    N, M = pos1.shape[1], pos2.shape[1]
    # error_uv12 = np.zeros(shape=(2, N, M), dtype=float)

    proj_uv12_ext = proj_uv12[:, :, None].repeat(M, axis=2)
    pos2_ext = pos2[:, None, :].repeat(N, axis=1)
    error_uv12 = proj_uv12_ext - pos2_ext
    error_uv12 = np.sqrt(np.sum(error_uv12 ** 2, axis=0))

    matches_12 = np.argmin(error_uv12, axis=1)
    errors_12 = error_uv12[np.arange(error_uv12.shape[0]), matches_12]

    inlier_ids12 = np.where(errors_12 <= inlier_th)
    outlier_ids12 = np.where(errors_12 > outlier_th)

    inlier_matches12 = np.array(np.vstack([inlier_ids12, matches_12[inlier_ids12]]), int).transpose()  # [N, 2]
    outlier_matches12 = np.array(np.vstack([outlier_ids12, matches_12[outlier_ids12]]), int).transpose()  # [N, 2]

    if not cycle_check:
        return inlier_matches12, outlier_matches12
    matched_pos1 = pos1[:, inlier_matches12[:, 0]]
    matched_pos2 = pos2[:, inlier_matches12[:, 1]]

    proj_uv21 = reproject(pos1=matched_pos2, depth1=depth2, intrinsics1=intrinsics2, pose1=pose2, bbox1=bbox2,
                          intrinsics2=intrinsics1, pose2=pose1, bbox2=bbox1)  # [2, M]
    error_uv21 = proj_uv21 - matched_pos1
    error_uv21 = np.sqrt(np.sum(error_uv21 ** 2, axis=0))
    inliers21 = (error_uv21 <= inlier_th)

    inlier_cycle = inlier_matches12[inliers21]
    outlier_cycle = np.vstack([inlier_matches12[~inliers21], outlier_matches12])

    return inlier_cycle, outlier_cycle


def match_from_projection_points(
        pos1, depth1, intrinsics1, pose1, bbox1,
        pos2, depth2, intrinsics2, pose2, bbox2,
        inlier_th=3, outlier_th=5,
        cycle_check=False):  # [2, M]

    proj_uv12 = reproject_points(pos1=pos1, depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=None,
                                 intrinsics2=intrinsics2, pose2=pose2, bbox2=None)  # [2, N]
    N, M = pos1.shape[1], pos2.shape[1]
    # error_uv12 = np.zeros(shape=(2, N, M), dtype=float)

    proj_uv12_ext = proj_uv12[:, :, None].repeat(M, axis=2)
    pos2_ext = pos2[:, None, :].repeat(N, axis=1)
    error_uv12 = proj_uv12_ext - pos2_ext
    error_uv12 = np.sqrt(np.sum(error_uv12 ** 2, axis=0))

    matches_12 = np.argmin(error_uv12, axis=1)
    errors_12 = error_uv12[np.arange(error_uv12.shape[0]), matches_12]

    inlier_ids12 = np.where(errors_12 <= inlier_th)
    outlier_ids12 = np.where(errors_12 > outlier_th)

    inlier_matches12 = np.array(np.vstack([inlier_ids12, matches_12[inlier_ids12]]), int).transpose()  # [N, 2]
    outlier_matches12 = np.array(np.vstack([outlier_ids12, matches_12[outlier_ids12]]), int).transpose()  # [N, 2]

    if not cycle_check:
        return inlier_matches12, outlier_matches12

    matched_pos1 = pos1[:, inlier_matches12[:, 0]]
    matched_pos2 = pos2[:, inlier_matches12[:, 1]]
    matched_depth2 = depth2[inlier_matches12[:, 1]]

    proj_uv21 = reproject_points(pos1=matched_pos2, depth1=matched_depth2, intrinsics1=intrinsics2, pose1=pose2,
                                 bbox1=bbox2,
                                 intrinsics2=intrinsics1, pose2=pose1, bbox2=bbox1)  # [2, M]
    error_uv21 = proj_uv21 - matched_pos1
    error_uv21 = np.sqrt(np.sum(error_uv21 ** 2, axis=0))
    inliers21 = (error_uv21 <= inlier_th)

    inlier_cycle = inlier_matches12[inliers21]
    outlier_cycle = np.vstack([inlier_matches12[~inliers21], outlier_matches12])

    return inlier_cycle, outlier_cycle


def projection_error_points_torch(pos1, depth1, intrinsics1, pose1, bbox1,
                                  pos2, depth2, intrinsics2, pose2, bbox2):
    proj_uv12 = reproject_points_torch(pos1=pos1, depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=None,
                                       intrinsics2=intrinsics2, pose2=pose2, bbox2=None)  # [2, N]
    N, M = pos1.shape[1], pos2.shape[1]
    # error_uv12 = np.zeros(shape=(2, N, M), dtype=float)

    proj_uv12_ext = proj_uv12[:, :, None].repeat(1, 1, M)
    pos2_ext = pos2[:, None, :].repeat(1, N, 1)
    error_uv12 = proj_uv12_ext - pos2_ext
    error_uv12 = torch.sqrt(torch.sum(error_uv12 ** 2, dim=0))

    return error_uv12


def match_from_projection_points_torch(
        pos1, depth1, intrinsics1, pose1, bbox1,
        pos2, depth2, intrinsics2, pose2, bbox2,
        inlier_th=3, outlier_th=5,
        cycle_check=False):  # [2, M]

    proj_uv12 = reproject_points_torch(pos1=pos1, depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=None,
                                       intrinsics2=intrinsics2, pose2=pose2, bbox2=None)  # [2, N]
    N, M = pos1.shape[1], pos2.shape[1]
    # error_uv12 = np.zeros(shape=(2, N, M), dtype=float)

    proj_uv12_ext = proj_uv12[:, :, None].repeat(1, 1, M)
    pos2_ext = pos2[:, None, :].repeat(1, N, 1)
    error_uv12 = proj_uv12_ext - pos2_ext
    error_uv12 = torch.sqrt(torch.sum(error_uv12 ** 2, dim=0))

    matches_12 = torch.argmin(error_uv12, dim=1)
    errors_12 = error_uv12[torch.arange(error_uv12.shape[0]), matches_12]

    inlier_ids12 = torch.where(errors_12 <= inlier_th)[0]
    outlier_ids12 = torch.where(errors_12 >= outlier_th)[0]

    inlier_matches12 = torch.vstack([inlier_ids12, matches_12[inlier_ids12]]).long().t()  # [N, 2]
    outlier_matches12 = torch.vstack([outlier_ids12, matches_12[outlier_ids12]]).long().t()  # [N, 2]

    if not cycle_check:
        return inlier_matches12, outlier_matches12

    matched_pos1 = pos1[:, inlier_matches12[:, 0]]
    matched_pos2 = pos2[:, inlier_matches12[:, 1]]
    matched_depth2 = depth2[inlier_matches12[:, 1]]

    proj_uv21 = reproject_points_torch(pos1=matched_pos2, depth1=matched_depth2, intrinsics1=intrinsics2, pose1=pose2,
                                       bbox1=bbox2,
                                       intrinsics2=intrinsics1, pose2=pose1, bbox2=bbox1)  # [2, M]
    error_uv21 = proj_uv21 - matched_pos1
    error_uv21 = torch.sqrt(torch.sum(error_uv21 ** 2, dim=0))
    inliers21 = (error_uv21 <= inlier_th)
    outlier21 = (error_uv21 >= outlier_th)

    inlier_cycle = inlier_matches12[inliers21]
    return inlier_cycle, outlier_matches12


def match_from_projection_points_torch_v2(
        pos1, depth1, intrinsics1, pose1, bbox1,
        pos2, depth2, intrinsics2, pose2, bbox2,
        inlier_th=3, outlier_th=5,
        cycle_check=False):  # [2, M]
    def compute(pos1, depth1, intrinsics1, pose1, bbox1,
                pos2, intrinsics2, pose2, bbox2,
                inlier_th=3, outlier_th=5):
        proj_uv12 = reproject_points_torch(pos1=pos1, depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=None,
                                           intrinsics2=intrinsics2, pose2=pose2, bbox2=None)  # [2, N]
        N, M = pos1.shape[1], pos2.shape[1]
        # error_uv12 = np.zeros(shape=(2, N, M), dtype=float)

        proj_uv12_ext = proj_uv12[:, :, None].repeat(1, 1, M)
        pos2_ext = pos2[:, None, :].repeat(1, N, 1)
        error_uv12 = proj_uv12_ext - pos2_ext
        error_uv12 = torch.sqrt(torch.sum(error_uv12 ** 2, dim=0))

        matches_12 = torch.argmin(error_uv12, dim=1)
        errors_12 = error_uv12[torch.arange(error_uv12.shape[0]), matches_12]
        inlier_ids12 = torch.where(errors_12 <= inlier_th)[0]
        outlier_ids12 = torch.where(errors_12 >= outlier_th)[0]

        inlier_matches12 = torch.vstack([inlier_ids12, matches_12[inlier_ids12]]).long().t()  # [N, 2]
        outlier_matches12 = torch.vstack([outlier_ids12, matches_12[outlier_ids12]]).long().t()  # [N, 2]

        return inlier_matches12, outlier_matches12

    inliers12, outlier12 = compute(pos1=pos1, depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=None,
                                   pos2=pos2, intrinsics2=intrinsics2, pose2=pose2, bbox2=None,
                                   inlier_th=inlier_th, outlier_th=outlier_th)
    inliers21, outlier21 = compute(pos1=pos2, depth1=depth2, intrinsics1=intrinsics2, pose1=pose2, bbox1=None,
                                   pos2=pos1, intrinsics2=intrinsics1, pose2=pose1, bbox2=None,
                                   inlier_th=inlier_th, outlier_th=outlier_th)

    return inliers12, inliers21, outlier12, outlier21


def match_from_projection_points_torch_v3(
        pos1, depth1, intrinsics1, pose1, bbox1,
        pos2, depth2, intrinsics2, pose2, bbox2,
        inlier_th=3, outlier_th=5,
        cycle_check=False):  # [2, M]
    def compute(pos1, depth1, intrinsics1, pose1, bbox1,
                pos2, intrinsics2, pose2, bbox2,
                inlier_th=3, outlier_th=5):
        proj_uv12 = reproject_points_torch(pos1=pos1, depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=None,
                                           intrinsics2=intrinsics2, pose2=pose2, bbox2=None)  # [2, N]
        N, M = pos1.shape[1], pos2.shape[1]
        # error_uv12 = np.zeros(shape=(2, N, M), dtype=float)

        proj_uv12_ext = proj_uv12[:, :, None].repeat(1, 1, M)
        pos2_ext = pos2[:, None, :].repeat(1, N, 1)
        error_uv12 = proj_uv12_ext - pos2_ext
        error_uv12 = torch.sqrt(torch.sum(error_uv12 ** 2, dim=0))

        matches_12 = torch.argmin(error_uv12, dim=1)
        errors_12 = error_uv12[torch.arange(error_uv12.shape[0]), matches_12]
        inlier_ids12 = torch.where(errors_12 <= inlier_th)[0]
        outlier_ids12 = torch.where(errors_12 >= outlier_th)[0]

        inlier_matches12 = torch.vstack([inlier_ids12, matches_12[inlier_ids12]]).long().t()  # [N, 2]
        outlier_matches12 = torch.vstack([outlier_ids12, matches_12[outlier_ids12]]).long().t()  # [N, 2]

        return inlier_matches12, outlier_matches12

    inliers12, outlier12 = compute(pos1=pos1, depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=None,
                                   pos2=pos2, intrinsics2=intrinsics2, pose2=pose2, bbox2=None,
                                   inlier_th=inlier_th, outlier_th=outlier_th)

    return inliers12, outlier12


def compute_epipolar_error(kpts0, kpts1, K0, K1, T_0to1, bbox0, bbox1):
    kpts0[:, 0] = kpts0[:, 0] + bbox0[0]
    kpts0[:, 1] = kpts0[:, 1] + bbox0[1]
    kpts1[:, 0] = kpts1[:, 0] + bbox1[0]
    kpts1[:, 1] = kpts1[:, 1] + bbox1[1]

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    print(T_0to1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    Etp1 = kpts1 @ E
    N = kpts0.shape[0]

    p1Ep0 = np.sum(kpts1 * Ep0, axis=1)
    ep_error = (p1Ep0 ** 2) * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
                               + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))

    return ep_error


def compute_eripolar_sampson_error(pts1, pts2, F):
    '''
    :param pts1: [N, 2]
    :param pts2: [N, 2]
    :param F: [3, 3] fundamental matrix
    :return:
    '''

    pts1_homo = np.concatenate([pts1, np.ones_like(pts1[:, :1])], axis=-1)
    pts2_homo = np.concatenate([pts2, np.ones_like(pts1[:, :1])], axis=-1)
    # print(pts1_homo.shape, pts2_homo.shape, F.shape)

    Fx1 = F @ pts1_homo.T  # [3, N]
    Fx2 = (pts2_homo @ F).T  # [3, N]
    demo = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    err = np.diag((pts2_homo @ (F @ pts1_homo.T))) ** 2 / demo  # sampson error
    # print('err: ', err)
    return err


def match_from_epipolar(kpts0, kpts1, K0, K1, T_0to1,
                        bbox0, bbox1,
                        inlier_th=3,
                        outlier_th=5):
    kpts0[:, 0] = kpts0[:, 0] + bbox0[0]
    kpts0[:, 1] = kpts0[:, 1] + bbox0[1]
    kpts1[:, 0] = kpts1[:, 0] + bbox1[0]
    kpts1[:, 1] = kpts1[:, 1] + bbox1[1]

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    print(T_0to1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    Etp1 = kpts1 @ E
    N, M = kpts0.shape[0], kpts1.shape[0]
    ep_error = np.zeros(shape=(N, M), dtype=float)
    for i in range(N):
        for j in range(M):
            # print(kpts1[j] * Ep0[i])
            p1Ep0 = np.sum(kpts1[j] * Ep0[i])
            d = (p1Ep0 ** 2) * (1.0 / (Ep0[i, 0] ** 2 + Ep0[i, 1] ** 2)
                                + 1.0 / (Etp1[j, 0] ** 2 + Etp1[j, 1] ** 2))
            ep_error[i, j] = d
            # print(i, j, d)

    matches = np.argmin(ep_error, axis=1)
    errors = ep_error[np.arange(ep_error.shape[0]), matches]
    matches_mutual = []
    unmatched_ids = []
    for i in range(N):
        print(i, np.min(ep_error[i]), np.median(ep_error[i]), np.max(ep_error[i]))
        if errors[i] > outlier_th:
            unmatched_ids.append(i)
            continue
        if errors[i] > inlier_th:
            continue

        id1 = matches[i]
        # if matches_21[id1] != i:  # no cycle consistency
        #     continue
        matches_mutual.append([i, id1])

    return np.array(matches_mutual, np.int)
    # p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    # Etp1 = kpts1 @ E  # N x 3
    # print('Ep1: ', Etp1.shape)
    # d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
    #                   + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x * 180 / math.pi, y * 180 / math.pi, z * 180 / math.pi])
