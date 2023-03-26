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
import os.path as osp
from copy import deepcopy
from nets.gm import normalize_keypoints
from tools.utils import plot_matches_spg, error_colormap
from tools.utils import compute_pose_error, angle_error_mat, angle_error_vec
from tools.common import resize_img
from components.utils.metrics import compute_epi_inlier
from components.utils.evaluation_utils import normalize_intrinsic


def plot_connections_one(img0, img1, pt, connected_pts, gt_cpt=None, show_text='', r=3):
    # print(connected_pts)
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    img_out = np.zeros(shape=(np.max([h0, h1]), w0 + w1, 3), dtype=np.uint8)
    # print(img0.shape, img1.shape, img_out.shape)
    img_out[:h0, :w0] = img0.copy()
    img_out[:h1, w0:] = img1.copy()

    for p in [pt]:
        img_out = cv2.circle(img_out, center=(int(p[0]), int(p[1])), radius=r, color=(0, 0, 255), thickness=3)
    for p in connected_pts:
        img_out = cv2.circle(img_out, center=(int(p[0] + w0), int(p[1])), radius=r, color=(0, 0, 255), thickness=3)

    ref_pts = [pt for i in range(len(connected_pts))]

    # color = (0, 0, 255)
    for id, (p0, p1) in enumerate(zip(ref_pts, connected_pts)):
        if id == 0:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        img_out = cv2.line(img_out, pt1=(int(p0[0]), int(p0[1])), pt2=(int(p1[0] + w0), int(p1[1])), color=color,
                           thickness=5)

    img_out = cv2.line(img_out, pt1=(int(ref_pts[0][0]), int(ref_pts[0][1])),
                       pt2=(int(connected_pts[0][0] + w0), int(connected_pts[0][1])), color=(0, 0, 255),
                       thickness=5)

    if gt_cpt is not None:
        # img_out = cv2.line(img_out, pt1=(int(pt[0]), int(pt[1])), pt2=(int(gt_cpt[0] + w0), int(gt_cpt[1])),
        #                    color=(255, 0, 0),
        #                    thickness=5)
        img_out = cv2.circle(img_out, center=(int(gt_cpt[0] + w0), int(gt_cpt[1])), radius=r * 2, color=(255, 0, 0),
                             thickness=3)
        img_out = cv2.circle(img_out, center=(int(gt_cpt[0] + w0), int(gt_cpt[1])), radius=r * 10, color=(255, 0, 0),
                             thickness=3)

    img_out = cv2.putText(img_out, '{:s}'.format(show_text), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    return img_out


def aug_matches_geo(matching_map, dist_map, pred_inliers0, pred_inliers1, gt_matches=None, error_th=20, topk=16,
                    score_th=-1, img0=None,
                    img1=None,
                    pts0=None, pts1=None, vis=False):
    if vis:
        wname = 'aug'
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    nh = 512
    mscores, matching_topks = torch.topk(torch.from_numpy(matching_map), k=topk, largest=True, dim=1)
    matching_topks = matching_topks.cpu().numpy()
    aug_matches = []
    aug_inliers = []
    for id0 in range(dist_map.shape[0]):
        if id0 in pred_inliers0:
            continue
        cans = np.where(dist_map[id0] <= error_th)[0]
        if cans.size == 0:
            continue

        mscore = matching_map[id0][cans]
        sorted_cans = cans[np.argsort(-mscore)]
        best_id1 = sorted_cans[0]

        if best_id1 in pred_inliers1:
            continue

        if score_th < 0:
            if matching_map[id0][best_id1] < 1. / topk * 2:
                continue
        else:
            if matching_map[id0][best_id1] < score_th:
                continue

        # print('mscore: ', matching_map[id0][sorted_cans])
        # print('dscore: ', dist_map[id0][sorted_cans])

        aug_matches.append((id0, best_id1))

        if vis:
            if gt_matches[id0] == best_id1:
                aug_inliers.append(True)
                gt_cpt = pts1[best_id1]
            else:
                aug_inliers.append(False)
                gt_cpt = None
            img = plot_connections_one(img0=img0, img1=img1, pt=pts0[id0], connected_pts=pts1[sorted_cans],
                                       show_text='Matching + Geo', gt_cpt=gt_cpt)

            cv2.imshow(wname, img)
            key = cv2.waitKey()
            if key in (27, ord('q')):
                continue
            # break
            # cv2.destroyAllWindows()
            # return

    aug_matches = np.array(aug_matches, dtype=int)

    if vis:
        print('aug matches by geo {:d}/{:d} inliers'.format(aug_matches.shape[0], np.sum(aug_inliers)))
        img_aug = plot_matches(img0=img0, img1=img1,
                               matches=aug_matches,
                               pts0=pts0,
                               pts1=pts1,
                               inliers=aug_inliers,
                               plot_outlier=True,
                               show_text='AUG:')
        img_aug = resize_img(img=img_aug, nh=nh)

        cv2.imshow(wname, img_aug)
        key = cv2.waitKey()
        if key in (27, ord('q')):
            # cv2.destroyAllWindows()
            return aug_matches

    return aug_matches


def _homo(x):
    # input: x [N, 2] or [batch_size, N, 2]
    # output: x_homo [N, 3]  or [batch_size, N, 3]
    assert len(x.size()) in [2, 3]
    # print(f"x: {x.size()[0]}, {x.size()[1]}, {x.dtype}, {x.device}")
    if len(x.size()) == 2:
        ones = torch.ones(x.size()[0], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 1)
    elif len(x.size()) == 3:
        ones = torch.ones(x.size()[0], x.size()[1], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 2)
    return x_homo


def _sampson_dist_general(F, X, Y, if_homo=False):
    if not if_homo:
        X = _homo(X)
        Y = _homo(Y)
    if len(X.size()) == 2:
        nominator = ((Y @ F @ X.t()) ** 2).t()
        Fx1 = torch.mm(F, X.t())
        Fx2 = torch.mm(F.t(), Y.t())
        # a = (Fx1[0:1, :] ** 2 + Fx1[1:2, :] ** 2).t().unsqueeze(2)
        # b = (Fx2[0:1, :] ** 2 + Fx2[1:2, :] ** 2).unsqueeze(0)
        # print(a.shape, b.shape)
        denom = (Fx2[0:1, :] ** 2 + Fx2[1:2, :] ** 2).t().unsqueeze(2) + (
                Fx1[0:1, :] ** 2 + Fx1[1:2, :] ** 2).unsqueeze(0)
        denom = denom[:, 0, :]
        errors = (nominator / denom).transpose()
    else:
        # nominator = (torch.diagonal(Y @ F @ X.transpose(1, 2), dim1=1, dim2=2)) ** 2  # [B, N]
        nominator = (Y @ F @ X.transpose(1, 2)) ** 2  # [B, M, N]
        Fx1 = torch.matmul(F, X.transpose(1, 2))  # [B, 3, 3] @ [B, 3, N] = [B, 3, N]
        Fx2 = torch.matmul(F.transpose(1, 2), Y.transpose(1, 2))  # [B, 3, 3] @ [B, 3, N] = [B, 3, N]
        denom = (Fx2[:, 0] ** 2 + Fx2[:, 1] ** 2).unsqueeze(2) + (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2).unsqueeze(1)
        # print(nominator.size(), denom.size(), F.shape, X.shape, Y.shape)
        errors = (nominator / denom).transpose(1, 2)
    return errors


def plot_matches(img0, img1, pts0, pts1, matches, inliers, line_thickness=1, scores=None, radius=3,
                 plot_outlier=True,
                 plot_kpts=False,
                 show_text=None):
    # print(matches.shape, len(inliers))
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    img_out = np.zeros(shape=(np.max([h0, h1]), w0 + w1, 3), dtype=np.uint8)
    img_out[:h0, :w0] = np.copy(img0)
    img_out[:h1, w0:] = np.copy(img1)

    if plot_kpts:
        for ip, p in enumerate(pts0):
            # if not inliers[ip]:
            #     continue
            if scores is not None:
                r = int(scores[ip] * radius)
            else:
                r = radius
            img_out = cv2.circle(img_out, center=(int(p[0]), int(p[1])), radius=r, color=(0, 0, 255),
                                 thickness=-1)
        for ip, p in enumerate(pts1):
            if scores is not None:
                r = int(scores[ip] * radius)
            else:
                r = radius
            img_out = cv2.circle(img_out, center=(int(p[0] + w0), int(p[1])), radius=r, color=(0, 0, 255),
                                 thickness=-1)
    for i in range(matches.shape[0]):
        color = (0, 0, 255)
        p0 = pts0[matches[i, 0]]
        p1 = pts1[matches[i, 1]]
        if inliers[i]:
            color = (0, 255, 0)
        else:
            if plot_outlier:
                color = (0, 0, 255)
            else:
                continue
        img_out = cv2.line(img_out, (int(p0[0]), int(p0[1])), (int(w0 + p1[0]), int(p1[1])),
                           color=color, thickness=line_thickness)
        img_out = cv2.circle(img_out, center=(int(p0[0]), int(p0[1])), radius=radius, color=(0, 0, 255),
                             thickness=2)
        img_out = cv2.circle(img_out, center=(int(p1[0] + w0), int(p1[1])), radius=radius, color=(0, 0, 255),
                             thickness=2)

    if show_text is not None:
        img_out = cv2.putText(img_out,
                              show_text,
                              (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 2,
                              (0, 0, 255), 3)

    return img_out


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def decompose_essestial_mat(E, pts0, pts1, K0, K1, distance_thresh=1000):
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


def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999, mask=None, method=cv2.RANSAC):
    # def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.999, mask=None, method=cv2.RANSAC):
    if len(kpts0) < 5:
        return None

    # '''
    # K = (K0 + K1) / 2.
    # trans_pts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    # trans_pts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    # trans_pts0 = cv2.undistortPoints(kpts0, cameraMatrix=K0, distCoeffs=None)[:, 0, :]
    # trans_pts1 = cv2.undistortPoints(kpts1, cameraMatrix=K1, distCoeffs=None)[:, 0, :]
    # trans_pts0[:, 0] = trans_pts0[:, 0] * K[0, 0] + K[0, 2]
    # trans_pts0[:, 1] = trans_pts0[:, 1] * K[1, 1] + K[1, 2]
    # trans_pts1[:, 0] = trans_pts1[:, 0] * K[0, 0] + K[0, 2]
    # trans_pts1[:, 1] = trans_pts1[:, 1] * K[1, 1] + K[1, 2]

    # E0, E0_mask = cv2.findEssentialMat(points1=trans_pts0,
    #                                    points2=trans_pts1,
    #                                    cameraMatrix=K,
    #                                    threshold=norm_thresh,
    #                                    prob=conf,
    #                                    mask=mask,
    #                                    method=method)
    # if E0 is None or E0.shape[0] != 3 or E0.shape[1] != 3:
    #     return None
    # '''
    # F0, E0_mask = cv2.findFundamentalMat(points1=kpts0,
    #                                      points2=kpts1,
    #                                      method=method,
    #                                      ransacReprojThreshold=3,
    #                                      confidence=conf)
    # E0 = K1.T @ F0 @ K0

    E1, E1_mask = cv2.findEssentialMat(points1=kpts0,
                                       points2=kpts1,
                                       cameraMatrix1=K0,
                                       cameraMatrix2=K1,
                                       distCoeffs1=None,
                                       distCoeffs2=None,
                                       threshold=norm_thresh,
                                       prob=conf,
                                       mask=mask,
                                       method=method)

    # E = E0
    # E_mask = E0_mask
    #
    E = E1
    E_mask = E1_mask

    if E is None or E.shape[0] != 3 or E.shape[1] != 3:
        return None

    # print('E0: ', E0)
    # print('E1: ', E1)

    R, t, mask_P = decompose_essestial_mat(E=E, pts0=kpts0[E_mask.ravel() > 0], pts1=kpts1[E_mask.ravel() > 0],
                                           K0=K0, K1=K1)

    mask = E_mask.ravel() >= 0
    mask[E_mask.ravel() > 0] = mask_P
    return E, R, t, E1_mask

    # ret = cv2.recoverPose(E, kpts0, kpts1, np.eye(3), mask=E_mask)
    # print('supp points: ', ret[0])
    # if E_mask.shape[0] > 5:
    # print('diff_mask: ', (E_mask.ravel() > 0) == (ret[3].ravel() > 0))
    # if ret is not None:
    #     return (E, ret[1], ret[2][:, 0], ret[3].ravel() > 0)
    # else:
    #     return None


def sample_pose_v2(pts0, pts1, mscore, K0, K1, n_minium=5, n_hypothesis=64, error_th=1, method=cv2.RANSAC):
    K = (K0 + K1) / 2
    # print('K: ', K.shape, K[0:2].shape, kpts0.shape, kpts0[None].shape)

    trans_pts0 = cv2.undistortPoints(pts0, cameraMatrix=K0, distCoeffs=None)[:, 0, :]
    trans_pts1 = cv2.undistortPoints(pts1, cameraMatrix=K1, distCoeffs=None)[:, 0, :]
    trans_pts0 = cv2.transform(trans_pts0[None], K[0:2])[0]
    trans_pts1 = cv2.transform(trans_pts1[None], K[0:2])[0]

    trans_pts0[:, 0] = (trans_pts0[:, 0] - K[0, 2]) / K[0, 0]
    trans_pts0[:, 1] = (trans_pts0[:, 1] - K[1, 2]) / K[1, 1]
    trans_pts1[:, 0] = (trans_pts1[:, 0] - K[0, 2]) / K[0, 0]
    trans_pts1[:, 1] = (trans_pts1[:, 1] - K[1, 2]) / K[1, 1]

    inlier_error_th = error_th / np.max([K[0, 0], K[1, 1]])

    sorted_ids = np.argsort(-mscore)
    best_E = None
    best_inlier = 0
    best_mask = None
    best_ip = -1
    best_error = 0
    for si in range(n_hypothesis):
        if si + n_minium >= sorted_ids.shape[0]:
            break
        sel_ids = sorted_ids[si: si + n_minium]
        # print(pts0.shape, pts1.shape, sel_ids.shape)

        if check_colinear(trans_pts0[sel_ids]) or check_colinear(trans_pts1[sel_ids]):
            continue

        # ret = estimate_pose(kpts0=pts0[sel_ids], kpts1=pts1[sel_ids], norm_thresh=error_th, conf=0.99999, K0=K0, K1=K1,
        #                     method=method)

        # E0, E0_mask = cv2.findEssentialMat(points1=pts0[sel_ids],
        #                                    points2=pts1[sel_ids],
        #                                    cameraMatrix1=K0,
        #                                    cameraMatrix2=K1,
        #                                    distCoeffs1=None,
        #                                    distCoeffs2=None,
        #                                    threshold=error_th,
        #                                    prob=0.99999,
        #                                    mask=None,
        #                                    method=method)

        E1, E1_mask = cv2.findEssentialMat(
            trans_pts0[sel_ids], trans_pts1[sel_ids], np.eye(3),
            threshold=inlier_error_th,
            prob=0.99999,
            method=method,
        )

        if E1 is None:
            continue
        # E, R, t, mask = ret

        # print('Pose-', si)
        # print('E0: ', E0)
        # print('E1: ', E1)

        E = E1
        proj_error = compute_reprojection_error(pts0=trans_pts0, pts1=trans_pts1, E=E)
        # print('proj_error: ', proj_error)
        # exit(0)
        n_inlier = np.sum(proj_error < inlier_error_th)
        error = np.median(proj_error[proj_error < inlier_error_th])

        if n_inlier > best_inlier:
            best_inlier = n_inlier
            best_E = E
            best_mask = proj_error < inlier_error_th
            best_ip = si
            best_error = error
        elif n_inlier == best_inlier and error < best_error:
            best_inlier = n_inlier
            best_E = E
            best_mask = proj_error < inlier_error_th
            best_ip = si
            best_error = error

    print('best It: {:d} inlier ratio: {:.2f}'.format(best_ip, best_inlier / pts0.shape[0]))
    R, t, mask_P = decompose_essestial_mat(E=best_E, pts0=pts0[best_mask], pts1=pts1[best_mask], K0=K0, K1=K1)

    return best_E, R, t, best_mask


def recover_pose_2(pts0, pts1, mscore, K0, K1, error_th, pose_hypothesis=0, method=cv2.RANSAC, use_refinement=True):
    # Estimate pose with all matches
    if pose_hypothesis == 0:
        ret = estimate_pose(pts0, pts1, K0, K1, error_th, method=method)
        return ret
    else:
        E, R, t, mask = sample_pose_v2(pts0=pts0, pts1=pts1,
                                       mscore=mscore,
                                       error_th=error_th,
                                       K0=K0,
                                       K1=K1,
                                       n_hypothesis=pose_hypothesis,
                                       n_minium=5,
                                       method=method)

        return E, R, t, mask


def compute_epipolar_error_T(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
                      + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))
    return d


def matching_iterative_v5(data, model, pose_model, nI=9, error_th=10, conf_th=0.95, inlier_th=0.5, match_ratio=0.2,
                          pose_hypothesis=0, stop_criteria={}, vis_matches=True, method=cv2.RANSAC, use_first=False,
                          topK=-1, aug_matches=False, use_refinement=True, min_kpts=0, save_root=None, save_fn=None):
    norm_pts0 = data['norm_pts0']
    norm_pts1 = data['norm_pts1']

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
    img_color0 = data['image_color0']
    img_color1 = data['image_color1']

    K0 = data['K0']
    K1 = data['K1']
    T_0to1 = data['T_0to1']

    last_best_R = None
    last_best_t = None

    if vis_matches:
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    valid_its = [3, 5, 7, 9, 11, 13, 14]

    for it in range(nI):
        with torch.no_grad():
            if it == 0:
                enc0, enc1 = model.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                                   scores1=scores1)
                desc0 = desc0 + enc0
                desc1 = desc1 + enc1
            desc0, desc1 = model.forward_one_layer(desc0=desc0, desc1=desc1, M0=None, M1=None, layer_i=it * 2)
            desc0, desc1 = model.forward_one_layer(desc0=desc0, desc1=desc1, M0=None, M1=None, layer_i=it * 2 + 1)

            ### only perform pose estimation at certain layers
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

            if topK > 0:
                sorted_ids = np.argsort(matched_score)[::-1][:topK]
                sorted_ids = sorted(sorted_ids)
                matched_ids0 = np.array(matched_ids0)[sorted_ids].tolist()
                matched_ids1 = np.array(matched_ids1)[sorted_ids].tolist()
                matched_score = np.array(matched_score)[sorted_ids].tolist()

            pred_matches = np.vstack([matched_ids0, matched_ids1]).transpose()
            mscore = np.array(matched_score)

            if pred_matches.shape[0] == 0:
                continue

            # gt_epi_errs = compute_epipolar_error_T(pts0_cpu[pred_matches[:, 0]], pts1_cpu[pred_matches[:, 1]],
            #                                        T_0to1, K0, K1)
            gt_epi_errs = compute_epipolar_error_T(pts0_cpu[pred_matches[:, 0]], pts1_cpu[pred_matches[:, 1]],
                                                   T_0to1, K0, K1)
            # print('gt_epi_error: ', np.min(gt_epi_errs), np.median(gt_epi_errs), np.max(gt_epi_errs))
            gt_inliers = (gt_epi_errs <= 0.0005)
            # ret = recover_pose(pts0=pts0_cpu[pred_matches[:, 0]], pts1=pts1_cpu[pred_matches[:, 1]],
            ret = recover_pose_2(pts0=pts0_cpu[pred_matches[:, 0]], pts1=pts1_cpu[pred_matches[:, 1]],
                                 K0=K0, K1=K1, error_th=error_th, mscore=mscore,
                                 pose_hypothesis=pose_hypothesis, method=method,
                                 use_refinement=use_refinement)

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

            if vis_matches:

                if save_fn is not None and save_root is not None:
                    # print(pred_matches.shape, pts0_cpu.shape, pose_inliers.reshape(-1).shape)
                    # print('pts0: ', pts0_cpu)
                    # print('inliers: ', pose_inliers)
                    correct = (pose_inliers.reshape(-1) > 0)
                    color_map = error_colormap(np.ones(shape=(np.sum(pose_inliers), 1)).reshape(-1))
                    plot_matches_spg(image0=cv2.cvtColor(img_color0, cv2.COLOR_BGR2RGB),
                                     image1=cv2.cvtColor(img_color1, cv2.COLOR_BGR2RGB),
                                     kpts0=pts0_cpu[pred_matches[:, 0], :][correct],
                                     kpts1=pts1_cpu[pred_matches[:, 1], :][correct],
                                     mkpts0=pts0_cpu[pred_matches[:, 0], :][correct],
                                     mkpts1=pts1_cpu[pred_matches[:, 1], :][correct],
                                     color=color_map,
                                     text=['error: {:.1f}R {:.1f}t'.format(error_R, error_t),
                                           'inliers: {:d}'.format(np.sum(pose_inliers))],
                                     save_path=osp.join(save_root, save_fn + '_{:02d}.png'.format(it)))
                img_match = plot_matches(img0=img_color0, img1=img_color1,
                                         pts0=pts0_cpu, pts1=pts1_cpu,
                                         matches=pred_matches,
                                         inliers=pose_inliers,
                                         # inliers=gt_inliers,
                                         plot_outlier=False,
                                         show_text='PI{:d}-{:.0f}/{:d}/{:.1f}/R:{:.1f},t:{:.1f}'.format(it,
                                                                                                        np.sum(
                                                                                                            pose_inliers),
                                                                                                        pred_matches.shape[
                                                                                                            0],
                                                                                                        np.sum(
                                                                                                            pose_inliers) /
                                                                                                        pred_matches.shape[
                                                                                                            0] if
                                                                                                        pred_matches.shape[
                                                                                                            0] > 0 else 0,
                                                                                                        error_R,
                                                                                                        error_t))

                cv2.imshow('img', img_match)
                cv2.waitKey(5)

            # Check if stop iteration
            if 'pose' in stop_criteria.keys():
                if pose_diff <= stop_criteria['pose']:
                    output_indice0 = np.zeros_like(indices0_cpu) - 1
                    output_indice0[pred_matches[pose_inliers, 0]] = pred_matches[pose_inliers, 1]

                    print('diff: ', diff_R, diff_t, np.sum(pose_inliers), pred_matches.shape[0])

                    if aug_matches:
                        pred_inlier_id0s = pred_matches[pose_inliers, 0].tolist()
                        pred_inlier_id1s = pred_matches[pose_inliers, 1].tolist()

                        kpts0_3d = (pts0_cpu - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
                        kpts1_3d = (pts1_cpu - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

                        proj_err_full = _sampson_dist_general(F=torch.from_numpy(E).cuda().float()[None],
                                                              X=torch.from_numpy(kpts0_3d).cuda().float()[None],
                                                              Y=torch.from_numpy(kpts1_3d).cuda().float()[None],
                                                              ).cpu().numpy()[0]
                        aug_matches = aug_matches_geo(matching_map=pred_score_cpu,
                                                      dist_map=proj_err_full,
                                                      gt_matches=None,
                                                      pred_inliers0=pred_inlier_id0s,
                                                      pred_inliers1=pred_inlier_id1s,
                                                      error_th=1 / np.mean(
                                                          [K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]]) * 1.5,
                                                      # error_th=error_th * 10,
                                                      score_th=0.2,
                                                      topk=16, )
                        if aug_matches.size > 0:
                            final_matches = np.vstack([aug_matches, pred_matches[pose_inliers]])
                        else:
                            final_matches = pred_matches[pose_inliers]

                        gt_epi_errs = compute_epipolar_error_T(pts0_cpu[final_matches[:, 0]],
                                                               pts1_cpu[final_matches[:, 1]],
                                                               T_0to1, K0, K1)
                        gt_accuracy = gt_epi_errs <= 1 / np.mean(
                            [K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])

                        if vis_matches:
                            img_aug = plot_matches(img0=img_color0, img1=img_color1,
                                                   pts0=pts0_cpu, pts1=pts1_cpu,
                                                   matches=final_matches,
                                                   inliers=gt_accuracy,
                                                   plot_outlier=False,
                                                   show_text='AugI{:d}-{:.0f}/{:d}/{:.1f}'.format(it,
                                                                                                  np.sum(
                                                                                                      gt_accuracy),
                                                                                                  final_matches.shape[
                                                                                                      0],
                                                                                                  np.sum(
                                                                                                      gt_accuracy) /
                                                                                                  final_matches.shape[
                                                                                                      0] if
                                                                                                  final_matches.shape[
                                                                                                      0] > 0 else 0))

                            cv2.imshow('img', np.vstack([img_match, img_aug]))
                            cv2.waitKey(0)

                    return output_indice0, mscores0_cpu, R, t, it + 1

    indices0, indices1, mscores0, mscores1 = model.compute_matches(scores=pred_score, p=0.2)
    indices0_cpu = indices0[0].cpu().numpy()
    mscores0_cpu = mscores0[0].cpu().numpy()

    return indices0_cpu, mscores0_cpu, None, None, nI
