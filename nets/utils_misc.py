# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> utils_misc
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   22/03/2022 17:42
=================================================='''
'''
code is from https://github.com/eric-yyjau/pytorch-deepFEPE/blob/012651c93f948cfd793cf8bba9670ab69abc0e04/deepFEPE/
'''
import torch
import numpy as np
from itertools import cycle
import cv2
import operator
import nets.utils_geo as utils_geo

cycol = cycle('bgrcmk')


def _gaussian_dist(xs, mean, var):
    prob = torch.exp(-(xs - mean) ** 2 / (2 * var))
    return prob


def within(x, y, xlim, ylim):
    val_inds = (x >= 0) & (y >= 0)
    val_inds = val_inds & (x <= xlim) & (y <= ylim)
    return val_inds


def identity_Rt(dtype=np.float32):
    return np.hstack((np.eye(3, dtype=dtype), np.zeros((3, 1), dtype=dtype)))


def _skew_symmetric(v):  # v: [3, 1] or [batch_size, 3, 1]
    if len(v.size()) == 2:
        zero = torch.zeros_like(v[0, 0])
        M = torch.stack([
            zero, -v[2, 0], v[1, 0],
            v[2, 0], zero, -v[0, 0],
            -v[1, 0], v[0, 0], zero,
        ], dim=0)
        return M.view(3, 3)
    else:
        zero = torch.zeros_like(v[:, 0, 0])
        M = torch.stack([
            zero, -v[:, 2, 0], v[:, 1, 0],
            v[:, 2, 0], zero, -v[:, 0, 0],
            -v[:, 1, 0], v[:, 0, 0], zero,
        ], dim=1)
        return M.view(-1, 3, 3)


def skew_symmetric_np(v):  # v: [3, 1] or [batch_size, 3, 1]
    if len(v.shape) == 2:
        zero = np.zeros_like(v[0, 0])
        M = np.stack([
            zero, -v[2, 0], v[1, 0],
            v[2, 0], zero, -v[0, 0],
            -v[1, 0], v[0, 0], zero,
        ], axis=0)
        return M.reshape(3, 3)
    else:
        zero = np.zeros_like(v[:, 0, 0])
        M = np.stack([
            zero, -v[:, 2, 0], v[:, 1, 0],
            v[:, 2, 0], zero, -v[:, 0, 0],
            -v[:, 1, 0], v[:, 0, 0], zero,
        ], axis=1)
        return M.reshape(-1, 3, 3)


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


def _de_homo(x_homo):
    # input: x_homo [N, 3] or [batch_size, N, 3]
    # output: x [N, 2] or [batch_size, N, 2]
    assert len(x_homo.size()) in [2, 3]
    epi = 1e-10
    if len(x_homo.size()) == 2:
        x = x_homo[:, :-1] / ((x_homo[:, -1] + epi).unsqueeze(-1))
    else:
        x = x_homo[:, :, :-1] / ((x_homo[:, :, -1] + epi).unsqueeze(-1))
    return x


def homo_np(x):
    # input: x [N, D]
    # output: x_homo [N, D+1]
    N = x.shape[0]
    x_homo = np.hstack((x, np.ones((N, 1), dtype=x.dtype)))
    return x_homo


def de_homo_np(x_homo):
    # input: x_homo [N, D]
    # output: x [N, D-1]
    assert x_homo.shape[1] in [3, 4]
    N = x_homo.shape[0]
    epi = 1e-10
    x = x_homo[:, :-1] / np.expand_dims(x_homo[:, -1] + epi, -1)
    return x


def Rt_pad(Rt):
    # Padding 3*4 [R|t] to 4*4 [[R|t], [0, 1]]
    assert Rt.shape == (3, 4)
    return np.vstack((Rt, np.array([[0., 0., 0., 1.]], dtype=Rt.dtype)))


# def _Rt_pad(Rt):
#     # Padding 3*4 [R|t] to 4*4 [[R|t], [0, 1]]
#     assert Rt.size()==(3, 4)
#     cat_tensor = torch.tensor([[0., 0., 0., 1.]], dtype=Rt.dtype)
#     return torch.cat((Rt, ))

def inv_Rt_np(Rt):
    assert Rt.shape == (3, 4)
    R1 = Rt[:, :3]
    t1 = Rt[:, 3:4]
    R2 = R1.T
    t2 = -R1.T @ t1
    return np.hstack((R2, t2))


def _inv_Rt(Rt):
    assert Rt.size() == (3, 4)
    R1 = Rt[:, :3]
    t1 = Rt[:, 3:4]
    R2 = R1.t()
    t2 = -R1.t() @ t1
    return torch.cat((R2, t2), 1)


def Rt_depad(Rt01):
    # dePadding 4*4 [[R|t], [0, 1]] to 3*4 [R|t]
    assert Rt01.shape == (4, 4)
    return Rt01[:3, :]


def vis_masks_to_inds(mask1, mask2):
    val_inds_both = mask1 & mask2
    val_idxes = [idx for idx in range(val_inds_both.shape[0]) if val_inds_both[idx]]  # within indexes
    return val_idxes


def normalize_Rt_to_1(Rt):
    assert Rt.shape == (4, 4) or Rt.shape == (3, 4)
    if Rt.shape == (4, 4):
        Rt = Rt[:3, :]
    return np.hstack((Rt[:, :3], Rt[:, 3:4] / (Rt[2, 3] + 1e-10)))


def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    # Adapted from https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93/fpointnet/data/datasets/utils.py
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points
    """
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, 'out_num_points = %d must be positive int!' % out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)
        choice = np.concatenate([choice, pad])
    return choice


def get_virt_x1x2_grid(im_shape):
    step = 0.1
    sz1 = im_shape
    sz2 = im_shape
    xx, yy = np.meshgrid(np.arange(0, 1, step), np.arange(0, 1, step))
    num_pts_full = len(xx.flatten())
    pts1_virt_b = np.float32(np.vstack((sz1[1] * xx.flatten(), sz1[0] * yy.flatten())).T)
    pts2_virt_b = np.float32(np.vstack((sz2[1] * xx.flatten(), sz2[0] * yy.flatten())).T)
    return pts1_virt_b, pts2_virt_b


def get_virt_x1x2_np(im_shape, F_gt, K, pts1_virt_b, pts2_virt_b):  ##  [RUI] TODO!!!!! Convert into seq loader!
    ## s.t. SHOULD BE ALL ZEROS: losses = utils_F.compute_epi_residual(pts1_virt_ori, pts2_virt_ori, F_gts, loss_params['clamp_at'])
    ## Reproject by minimizing distance to groundtruth epipolar lines
    pts1_virt, pts2_virt = cv2.correctMatches(F_gt, np.expand_dims(pts2_virt_b, 0), np.expand_dims(pts1_virt_b, 0))
    pts1_virt[np.isnan(pts1_virt)] = 0.
    pts2_virt[np.isnan(pts2_virt)] = 0.

    # nan1 = np.logical_and(
    #         np.logical_not(np.isnan(pts1_virt[:,:,0])),
    #         np.logical_not(np.isnan(pts1_virt[:,:,1])))
    # nan2 = np.logical_and(
    #         np.logical_not(np.isnan(pts2_virt[:,:,0])),
    #         np.logical_not(np.isnan(pts2_virt[:,:,1])))
    # _, midx = np.where(np.logical_and(nan1, nan2))
    # good_pts = len(midx)
    # while good_pts < num_pts_full:
    #     midx = np.hstack((midx, midx[:(num_pts_full-good_pts)]))
    #     good_pts = len(midx)
    # midx = midx[:num_pts_full]
    # pts1_virt = pts1_virt[:,midx]
    # pts2_virt = pts2_virt[:,midx]

    pts1_virt = homo_np(pts1_virt[0])
    pts2_virt = homo_np(pts2_virt[0])
    pts1_virt_normalized = (np.linalg.inv(K) @ pts1_virt.T).T
    pts2_virt_normalized = (np.linalg.inv(K) @ pts1_virt.T).T
    return pts1_virt_normalized, pts2_virt_normalized, pts1_virt, pts2_virt


def get_virt_x1x2(im_shape, F_gt, K, pts1_virt_b=None, pts2_virt_b=None):  ##  [RUI] TODO!!!!! Convert into seq loader!
    ## s.t. SHOULD BE ALL ZEROS: losses = utils_F.compute_epi_residual(pts1_virt_ori, pts2_virt_ori, F_gts, loss_params['clamp_at'])
    if pts1_virt_b is None and pts2_virt_b is None:
        pts1_virt_b, pts2_virt_b = get_virt_x1x2_grid(im_shape)
    ## Reproject by minimizing distance to groundtruth epipolar lines
    pts1_virt, pts2_virt = cv2.correctMatches(F_gt, np.expand_dims(pts2_virt_b, 0), np.expand_dims(pts1_virt_b, 0))
    pts1_virt[np.isnan(pts1_virt)] = 0.
    pts2_virt[np.isnan(pts2_virt)] = 0.

    # nan1 = np.logical_and(
    #         np.logical_not(np.isnan(pts1_virt[:,:,0])),
    #         np.logical_not(np.isnan(pts1_virt[:,:,1])))
    # nan2 = np.logical_and(
    #         np.logical_not(np.isnan(pts2_virt[:,:,0])),
    #         np.logical_not(np.isnan(pts2_virt[:,:,1])))
    # _, midx = np.where(np.logical_and(nan1, nan2))
    # good_pts = len(midx)
    # while good_pts < num_pts_full:
    #     midx = np.hstack((midx, midx[:(num_pts_full-good_pts)]))
    #     good_pts = len(midx)
    # midx = midx[:num_pts_full]
    # pts1_virt = pts1_virt[:,midx]
    # pts2_virt = pts2_virt[:,midx]

    pts1_virt = homo_np(pts1_virt[0])
    pts2_virt = homo_np(pts2_virt[0])
    pts1_virt_normalized = (np.linalg.inv(K) @ pts1_virt.T).T
    pts2_virt_normalized = (np.linalg.inv(K) @ pts1_virt.T).T
    return torch.from_numpy(pts1_virt_normalized).float(), torch.from_numpy(pts2_virt_normalized).float(), \
           torch.from_numpy(pts1_virt).float(), torch.from_numpy(pts2_virt).float()


def _prob_sampling(matches: torch.Tensor, n_minium: int, n_hypothesis: int):
    '''
    :param matches: [N, M]
    :param minium_n: int
    :param n_hypothesis: int
    :param probability: [N, M]
    :return:
    '''

    def sample(ids, selected_ids):
        sampled_ids = []
        sampled_irs = []
        for i in ids:
            ir = i // M
            # print('i, ir: ', i, ir)
            if i in selected_ids:
                continue
            if i in sampled_ids:
                continue
            if ir in sampled_irs:
                continue

            sampled_ids.append(i)
            sampled_irs.append(ir)

            if len(sampled_ids) == n_minium:
                return sampled_ids

        return None

    N = matches.shape[0]
    M = matches.shape[1]
    # print('NM: ', N, M)
    selected_ids = []
    output_pairs = []
    all_pairs = matches.contiguous().view(-1)
    sorted_ids = torch.argsort(-all_pairs)  # from small to large
    # print('sorted_ids: ', sorted_ids.shape)

    for nt in range(n_hypothesis):
        sampled_ids = sample(ids=sorted_ids, selected_ids=selected_ids)

        if sampled_ids is None:
            print('Not enough samples')
            break

        id_xs = []
        id_ys = []
        for sid in sampled_ids:
            selected_ids.append(sid)
            id_ys.append(sid // M)
            id_xs.append(sid % M)
        output_pairs.append(
            torch.vstack([torch.tensor(id_ys, dtype=torch.long, device=matches.device),
                          torch.tensor(id_xs, dtype=torch.long, device=matches.device)])[None])

    if len(output_pairs) == 1:
        return output_pairs
    else:
        return torch.vstack(output_pairs)


def _prob_sampling_simple(matches: torch.Tensor, n_minium: int, n_hypothesis: int):
    '''
        :param matches: [N, M]
        :param minium_n: int
        :param n_hypothesis: int
        :param probability: [N, M]
        :return:
        '''
    N = matches.shape[0]
    M = matches.shape[1]
    all_pairs = matches.contiguous().view(-1)
    sorted_ids = torch.argsort(-all_pairs)  # from small to large
    # print('sorted_ids: ', sorted_ids.shape)

    sampled_ids = sorted_ids[:n_hypothesis * n_minium]
    id_ys = (sampled_ids // M).reshape(n_hypothesis * n_minium, 1)
    id_xs = (sampled_ids % M).reshape(n_hypothesis * n_minium, 1)

    return torch.cat([id_ys, id_xs], dim=-1)


def _prob_sampling_simple_seq(matches: torch.Tensor, n_minium: int, n_hypothesis: int):
    '''
        :param matches: [N, M]
        :param minium_n: int
        :param n_hypothesis: int
        :param probability: [N, M]
        :return:
        '''
    N = matches.shape[0]
    M = matches.shape[1]
    all_pairs = matches.contiguous().view(-1)
    sorted_ids = torch.argsort(-all_pairs)  # from small to large
    # print('sorted_ids: ', sorted_ids.shape)

    sampled_ids = torch.hstack([sorted_ids[i:i + n_minium] for i in range(n_hypothesis)])
    # print(sampled_ids, sampled_ids.shape)
    id_ys = (sampled_ids // M).reshape(n_hypothesis * n_minium, 1)
    id_xs = (sampled_ids % M).reshape(n_hypothesis * n_minium, 1)

    return torch.cat([id_ys, id_xs], dim=-1)


def _prob_sampling_simple_seq_v2(matches: torch.Tensor, n_minium: int, n_hypothesis: int):
    '''
        :param matches: [N, M]
        :param minium_n: int
        :param n_hypothesis: int
        :param probability: [N, M]
        :return:
        '''
    N = matches.shape[0]
    M = matches.shape[1]
    values, full_xs = torch.max(matches, dim=1)
    # all_pairs = matches.contiguous().view(-1)
    sorted_ids = torch.argsort(-values)  # from large to small
    # print('sorted_ids: ', sorted_ids.shape)

    id_ys = torch.hstack([sorted_ids[i:i + n_minium] for i in range(n_hypothesis)]).reshape(-1, 1)
    id_xs = torch.hstack([full_xs[sorted_ids[i:i + n_minium]] for i in range(n_hypothesis)]).reshape(-1, 1)

    return torch.cat([id_ys, id_xs], dim=-1)


def _prob_sampling_simple_v2(matches: torch.Tensor, n_minium: int, n_hypothesis: int):
    '''
        :param matches: [N, M]
        :param minium_n: int
        :param n_hypothesis: int
        :param probability: [N, M]
        :return:
        '''
    N = matches.shape[0]
    M = matches.shape[1]
    values, full_xs = torch.max(matches, dim=1)
    # all_pairs = matches.contiguous().view(-1)
    sorted_ids = torch.argsort(-values)  # from small to large
    # print('sorted_ids: ', sorted_ids.shape)

    id_ys = sorted_ids[:n_hypothesis * n_minium].reshape(-1, 1)
    id_xs = full_xs[sorted_ids[:n_hypothesis * n_minium]].reshape(-1, 1)

    return torch.cat([id_ys, id_xs], dim=-1)


def _F_to_E(F, K1, K2):
    if len(F.size()) == 2:
        E = K2.transpose(0, 1) @ F @ K1
    else:
        E = torch.matmul(K2.transpose(1, 2), torch.matmul(F, K1))

    # print('torch F_to_E: ', E)
    return E


def _E_to_F(E, K1, K2):
    if len(E.size()) == 2:
        return torch.inverse(K2).transpose(0, 1) @ E @ torch.inverse(K1)
    else:
        K2_inv = torch.inverse(K2)
        K1_inv = torch.inverse(K1)

        return torch.matmul(K2_inv.transpose(1, 2), torch.matmul(E, K1_inv))


def _get_M2s(E):
    # Getting 4 possible poses from E
    U, S, V = torch.svd(E)
    W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=E.dtype, device=E.device)
    if torch.det(torch.mm(U, torch.mm(W, V.t()))) < 0:
        W = -W
    # print('-- delta_t_gt', delta_t_gt)

    t_recover = U[:, 2:3]  # / torch.norm(U[:, 2:3])
    # print('---', E.numpy())
    # t_recover_rescale = U[:, 2]/torch.norm(U[:, 2])*np.linalg.norm(t_gt) # -t_recover_rescale is also an option
    R_recover_1 = torch.mm(U, torch.mm(W, V.t()))
    R_recover_2 = torch.mm(U, torch.mm(W.t(), V.t()))  # also an option
    # print('-- t_recover', t_recover.numpy())
    # print('-- R_recover_1', R_recover_1.numpy(), torch.det(R_recover_1).numpy())
    # print('-- R_recover_2', R_recover_2.numpy(), torch.det(R_recover_2).numpy())

    R2s = [R_recover_1, R_recover_2]
    t2s = [t_recover, -t_recover]
    M2s = [torch.cat((x, y), 1) for x, y in [(x, y) for x in R2s for y in t2s]]
    return R2s, t2s, M2s


def _decomposeE(E):
    U, D, Vt = torch.linalg.svd(E)
    # print('torch D: ', D)
    # print('torch U: ', U)
    # print('torch Vt: ', Vt)
    if torch.det(U) < 0:
        U *= -1
    if torch.det(Vt) < 0:
        Vt *= -1

    W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=E.dtype, device=E.device)

    R1 = U @ W @ Vt
    R2 = U @ W.transpose(0, 1) @ Vt
    t = U[:, 2:3]
    R2s = [R1, R2]
    t2s = [t, -t]
    M2s = [torch.cat((x, y), 1) for x, y in [(x, y) for x in R2s for y in t2s]]
    return R2s, t2s, M2s


def _recoverPose(E: torch.Tensor, x1: np.ndarray, x2: np.ndarray, mask=None, depth_dist=50):
    def within_mask(Z, thres_min, thres_max):
        return (Z > thres_min) & (Z < thres_max)

    Rs, ts, Ms = _decomposeE(E=E)
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    M1 = np.hstack((R1, t1))
    if mask is not None:
        x1 = x1[mask]
        x2 = x2[mask]
        if x1.shape[0] <= 8:
            print('Not enough points for triangulation')

    all_masks = []
    # triangulated_points = []
    K = np.eye(3, 3, dtype=x1.dtype)
    for id, M2 in enumerate(Ms):
        M2 = M2.detach().cpu().numpy()
        R2 = M2[:, 0:3]
        t2 = M2[:, 3:4]

        tri_x_homo = cv2.triangulatePoints(np.matmul(K, M1), np.matmul(K, M2), x1.T, x2.T)  # [4, N]
        tri_x = tri_x_homo[:3, :] / tri_x_homo[-1, :]

        mask_1 = within_mask(tri_x[-1, :], 0., depth_dist)
        tri_x_2 = np.matmul(R2, tri_x) + t2
        mask_2 = within_mask(tri_x_2[-1, :], 0., depth_dist)
        mask_12 = mask_1 & mask_2
        all_masks.append(mask_12)

    good_M_index, non_zero_nums = max(enumerate([np.sum(mask) for mask in all_masks]),
                                      key=operator.itemgetter(1))
    if non_zero_nums > 0:
        R = Ms[good_M_index][:, :3]
        t = Ms[good_M_index][:, 3:4]
        return R, t
    else:
        print('failed to find a good [R|t]')
        return None, None


def _E_to_M_train(E_est_th, K1, K2, x1, x2, inlier_mask=None, delta_Rt_gt_cam=None, depth_thres=50., show_debug=False,
                  show_result=True, method_name='ours'):
    if show_debug:
        print('--- Recovering pose from E...')
    count_N = x1.shape[0]
    # R2s, t2s, M2s = _get_M2s(E_est_th)
    R2s, t2s, M2s = _decomposeE(E_est_th)

    # print('torch Rs: ', R2s)
    # print('torch ts: ', t2s)

    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    M1 = np.hstack((R1, t1))

    if inlier_mask is not None:
        x1 = x1[inlier_mask, :]
        x2 = x2[inlier_mask, :]
        if x1.shape[0] < 8:
            print('ERROR! Less than 8 points after inlier mask!')
            print(inlier_mask)
            return None
    # Cheirality check following OpenCV implementation: https://github.com/opencv/opencv/blob/808ba552c532408bddd5fe51784cf4209296448a/modules/calib3d/src/five-point.cpp#L513
    depth_thres = depth_thres
    cheirality_checks = []
    M2_list = []

    def within_mask(Z, thres_min, thres_max):
        return (Z > thres_min) & (Z < thres_max)

    for Rt_idx, M2 in enumerate(M2s):
        M2 = M2.detach().cpu().numpy()
        R2 = M2[:, :3]
        t2 = M2[:, 3:4]
        if show_debug:
            print('M2: ', M2)
            print('det(R2)', np.linalg.det(R2))
            print('K-M: ', K1.shape, K2.shape, M1.shape, M2.shape, x1.shape, x2.shape)

        X_tri_homo = cv2.triangulatePoints(np.matmul(K1, M1), np.matmul(K2, M2), x1.T, x2.T)  # [4, N]
        X_tri = X_tri_homo[:3, :] / X_tri_homo[-1, :]
        # C1 = -np.matmul(R1, t1) # https://math.stackexchange.com/questions/82602/how-to-find-camera-position-and-rotation-from-a-4x4-matrix
        # cheirality1 = np.matmul(R1[2:3, :], (X_tri-C1)).reshape(-1) # https://cmsc426.github.io/sfm/
        # if show_debug:
        #     print(X_tri[-1, :])
        cheirality_mask_1 = within_mask(X_tri[-1, :], 0., depth_thres)

        X_tri_cam2 = np.matmul(R2, X_tri) + t2
        # C2 = -np.matmul(R2, t2)
        # cheirality2 = np.matmul(R2[2:3, :], (X_tri_cam3-C2)).reshape(-1)
        cheirality_mask_2 = within_mask(X_tri_cam2[-1, :], 0., depth_thres)

        cheirality_mask_12 = cheirality_mask_1 & cheirality_mask_2
        cheirality_checks.append(cheirality_mask_12)

        # print('R-mask: ', R2, t2, np.sum(cheirality_mask_12))

    if show_debug:
        print([np.sum(mask) for mask in cheirality_checks])
    good_M_index, non_zero_nums = max(enumerate([np.sum(mask) for mask in cheirality_checks]),
                                      key=operator.itemgetter(1))
    if non_zero_nums > 0:
        # Rt_idx = cheirality_checks.index(True)
        # M_inv = utils_misc.Rt_depad(np.linalg.inv(utils_misc.Rt_pad(M2s[good_M_index].detach().cpu().numpy())))
        # M_inv = utils_misc.inv_Rt_np(M2s[good_M_index].detach().cpu().numpy())

        # M_inv_th = _inv_Rt(M2s[good_M_index])
        M_inv_th = M2s[good_M_index]
        # print(M_inv, M_inv_th)
        if show_debug:
            print('The %d_th (0-based) Rt meets the Cheirality Condition! with [R|t] (camera):\n' % good_M_index,
                  M_inv_th.detach().cpu().numpy())

        if delta_Rt_gt_cam is not None:
            # R2 = M2s[good_M_index][:, :3].numpy()
            # t2 = M2s[good_M_index][:, 3:4].numpy()
            # error_R = min([utils_geo.rot12_to_angle_error(R2.numpy(), delta_R_gt) for R2 in R2s])
            # error_t = min(utils_geo.vector_angle(t2, delta_t_gt), utils_geo.vector_angle(-t2, delta_t_gt))
            M_inv = M_inv_th.detach().cpu().numpy()
            R2 = M_inv[:, :3]
            t2 = M_inv[:, 3:4]
            error_R = utils_geo.rot12_to_angle_error(R2, delta_Rt_gt_cam[:3, :3])  # [RUI] Both of camera motion
            error_t = utils_geo.vector_angle(t2, delta_Rt_gt_cam[:3, 3:4])
            if show_result:
                print(
                    'Recovered by %s (camera): The rotation error (degree) %.4f, and translation error (degree) %.4f' % (
                        method_name, error_R, error_t))
            error_Rt = [error_R, error_t]
        else:
            error_Rt = []
        Rt_cam = M_inv_th

    else:
        # raise ValueError('ERROR! 0 of qualified [R|t] found!')
        print('ERROR! 0 of qualified [R|t] found!')
        error_Rt = []
        Rt_cam = None

    return M2_list, error_Rt, Rt_cam


def _sampson_dist(F, X, Y, if_homo=False):
    if not if_homo:
        X = _homo(X)
        Y = _homo(Y)
    if len(X.size()) == 2:
        nominator = (torch.diag(Y @ F @ X.t())) ** 2
        Fx1 = torch.mm(F, X.t())
        Fx2 = torch.mm(F.t(), Y.t())
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    else:
        nominator = (torch.diagonal(Y @ F @ X.transpose(1, 2), dim1=1, dim2=2)) ** 2
        Fx1 = torch.matmul(F, X.transpose(1, 2))
        Fx2 = torch.matmul(F.transpose(1, 2), Y.transpose(1, 2))
        denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Fx2[:, 0] ** 2 + Fx2[:, 1] ** 2
        # print(nominator.size(), denom.size())

    errors = nominator / denom
    return errors


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


def _sym_epi_dist(F, X, Y, if_homo=False, clamp_at=None):
    # Actually squared
    if not if_homo:
        X = _homo(X)
        Y = _homo(Y)
    if len(X.size()) == 2:
        nominator = (torch.diag(Y @ F @ X.t())) ** 2
        Fx1 = torch.mm(F, X.t())
        Fx2 = torch.mm(F.t(), Y.t())
        denom_recp = 1. / (Fx1[0] ** 2 + Fx1[1] ** 2) + 1. / (Fx2[0] ** 2 + Fx2[1] ** 2)
    else:
        # print('-', X.detach().cpu().numpy())
        # print('-', Y.detach().cpu().numpy())
        # print('--', F.detach().cpu().numpy())
        nominator = (torch.diagonal(Y @ F @ X.transpose(1, 2), dim1=1, dim2=2)) ** 2
        Fx1 = torch.matmul(F, X.transpose(1, 2))
        # print(Fx1.detach().cpu().numpy(), torch.max(Fx1), torch.sum(Fx1))
        # print(X.detach().cpu().numpy(), torch.max(X), torch.sum(X))
        Fx2 = torch.matmul(F.transpose(1, 2), Y.transpose(1, 2))
        denom_recp = 1. / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + 1e-10) + 1. / (Fx2[:, 0] ** 2 + Fx2[:, 1] ** 2 + 1e-10)
        # print(nominator.size(), denom.size())

    errors = nominator * denom_recp
    # print('---', nominator.detach().cpu().numpy())
    # print('---------', denom_recp.detach().cpu().numpy())

    if clamp_at is not None:
        errors = torch.clamp(errors, max=clamp_at)

    return errors


def _error_proj_H(H, X, Y, if_homo=False):
    if not if_homo:
        X = _homo(X)
    if len(X.size()) == 2:
        X_warp = (H @ X.t()).t()
        X_warp = _de_homo(X_warp)
        err = torch.sqrt(torch.sum((Y - X_warp) ** 2, dim=1))
    else:
        X_warp = torch.bmm(H, X.transpose(1, 2)).transpose(1, 2)
        X_warp = _de_homo(X_warp)
        err = torch.sqrt(torch.sum((Y - X_warp) ** 2, dim=2))
    return err


def _error_proj_H_general(H, X, Y, if_homo=False):
    if not if_homo:
        X = _homo(X)
    if len(X.size()) == 2:
        X_warp = (H @ X.t()).t()
        X_warp = _de_homo(X_warp)
        err = torch.sqrt(
            torch.sum((X_warp.unsqueeze(2) - Y.t().unsqueeze(0)) ** 2, dim=1))  # [N, 2, 1] - [1, 2, M] = [N, 2, M]
    else:
        X_warp = torch.bmm(H, X.transpose(1, 2)).transpose(1, 2)  # [B, N, 3]
        X_warp = _de_homo(X_warp)  # [B, N, 2]
        err = torch.sqrt(torch.sum((X_warp.unsqueeze(3) - Y.transpose(1, 2).unsqueeze(1)) ** 2,
                                   dim=2))  # [B, N, 2, 1] - [B, 1, 2, M] = [B, N, 2, M]
    return err


# def _get_M2s_batch(Es_batch):
#     from batch_svd import batch_svd  # https://github.com/KinglittleQ/torch-batch-svd.git
#     # Getting 4 possible poses from E
#     Us, Ss, Vs = batch_svd(Es_batch)
#     Ws_list = []
#     for U, V in zip(Us, Vs):
#         W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=Es_batch.dtype, device=Es_batch.device)
#         if torch.det(torch.mm(U, torch.mm(W, V.t()))) < 0:
#             W = -W
#         Ws_list.append(W)
#     Ws = torch.stack(Ws_list)
#
#     t_recover_batch = Us[:, :, 2:3] / torch.norm(Us[:, :, 2:3], p=2, dim=1, keepdim=True)
#     R_recover_1_batch = Us @ Ws @ Vs.transpose(1, 2)
#     R_recover_2_batch = Us @ Ws.transpose(1, 2) @ Vs.transpose(1, 2)  # also an option
#
#     R2s_batch = [R_recover_1_batch, R_recover_2_batch]
#     t2s_batch = [t_recover_batch, -t_recover_batch]
#     # M2s = [torch.cat((x, y), 1) for x, y in [(x,y) for x in R2s for y in t2s]]
#     return R2s_batch, t2s_batch


def recover_pose_from_F(pts1, pts2, F, K1, K2, mask=None):
    '''
    :param pts1: [N, 2]
    :param pts2: [N, 2]
    :param F: [3, 3]
    :param K1: [3, 3]
    :param K2: [3, 3]
    :return: R, t
    '''
    # F = k1.inv().t @ E @ K1.inv()
    E = K2.T @ F @ K1
    # print('cv2: E_from_F: ', E)

    # R1, R2, t = cv2.decomposeEssentialMat(E=E)
    # print('2Rs: ', R1, R2)
    # print('ts: ', t)
    # D, U, Vt = cv2.SVDecomp(E)
    # print('cv2: D', D)
    # print('cv2: U', U)
    # print('cv2: Vt', Vt)

    # R, t = None, None
    success, R, t, mask = cv2.recoverPose(E=E, points1=pts1, points2=pts2, mask=mask)
    return R, -t


def epipolar_error(pts1, pts2, F):
    pts1_homo = np.concatenate([pts1, np.ones_like(pts1[:, :1])], axis=-1)
    pts2_homo = np.concatenate([pts2, np.ones_like(pts1[:, :1])], axis=-1)

    Fx1 = F @ pts1_homo.T  # [3, N]
    Fx2 = (pts2_homo @ F).T  # [3, N]
    demo = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    err = np.diag((pts2_homo @ (F @ pts1_homo.T))) ** 2 / demo  # sampson error
    return err


def epipolar_error_full(pts1, pts2, F):
    pts1_homo = np.concatenate([pts1, np.ones_like(pts1[:, :1])], axis=-1)
    pts2_homo = np.concatenate([pts2, np.ones_like(pts1[:, :1])], axis=-1)

    Fx1 = F @ pts1_homo.T  # [3, N]
    Fx2 = (pts2_homo @ F).T  # [3, N]
    demo = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    err = np.diag((pts2_homo @ (F @ pts1_homo.T))) ** 2 / demo  # sampson error
    return err


def projection_error_homography(pts1, pts2, H):
    pts1_homo = np.concatenate([pts1, np.ones_like(pts1[:, :1])], axis=-1)
    # pts2_homo = np.concatenate([pts2, np.ones_like(pts1[:, :1])], axis=-1)

    pts1_warp = H @ pts1_homo.T  # [3, N]
    pts1_warp[0, :] = pts1_warp[0, :] / pts1_warp[2, :]
    pts1_warp[1, :] = pts1_warp[1, :] / pts1_warp[2, :]

    err = np.sqrt(np.sum((pts2 - pts1_warp[:2, :].T) ** 2, axis=1))
    return err


def projection_error_homography_full(pts1, pts2, H):
    pts1_homo = np.concatenate([pts1, np.ones_like(pts1[:, :1])], axis=-1)
    pts1_warp = H @ pts1_homo.T  # [3, N]
    pts1_warp[0, :] = pts1_warp[0, :] / pts1_warp[2, :]
    pts1_warp[1, :] = pts1_warp[1, :] / pts1_warp[2, :]
    pts1_warp = np.repeat(pts1_warp, pts2.shape[0], axis=1)  # [3, MN]

    pts2_exp = np.tile(pts2, (pts1.shape[0], 1))  # [MN, 2]

    err = np.sqrt(np.sum((pts1_warp[:2, :].T - pts2_exp) ** 2, axis=1))  # [MN,1]
    err = err.reshape(pts1.shape[0], pts2.shape[0])
    return err


def position_error(t1, t2):
    diff = np.array(t1) - np.array(t2)
    return np.sqrt(np.sum(diff ** 2))


def angular_error(q1, q2):
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


def prob_sampling(matches, n_minium, n_hypothesis):
    '''
    :param matches: [N, M]
    :param minium_n: int
    :param n_hypothesis: int
    :param probability: [N, M]
    :return:
    '''

    def sample(ids, selected_ids):
        sampled_ids = []
        sampled_irs = []
        for i in ids:
            ir = i // M
            # print('i, ir: ', i, ir)
            if i in selected_ids:
                continue
            if i in sampled_ids:
                continue
            if ir in sampled_irs:
                continue

            sampled_ids.append(i)
            sampled_irs.append(ir)

            if len(sampled_ids) == n_minium:
                return sampled_ids

        return None

    N = matches.shape[0]
    M = matches.shape[1]
    selected_ids = []
    all_pairs = matches.reshape(-1)
    sorted_ids = np.argsort(-all_pairs)  # from small to large
    # print('sorted_ids: ', sorted_ids.shape)

    output_pairs = []
    # output_scores = []

    for nt in range(n_hypothesis):
        sampled_ids = sample(ids=sorted_ids, selected_ids=selected_ids)

        if sampled_ids is None:
            print('Not enough samples')
            break

        id_xs = []
        id_ys = []
        scores = []
        for sid in sampled_ids:
            selected_ids.append(sid)
            id_ys.append(sid // M)
            id_xs.append(sid % M)
            scores.append(all_pairs[sid])
        output_pairs.append(
            np.vstack([np.array(id_ys, dtype=int),
                       np.array(id_xs, dtype=int)])[None])
    if len(output_pairs) == 1:
        return np.array(output_pairs)
    else:
        return np.vstack(output_pairs)


def estimate_H(pts1, pts2, method=-1):
    assert pts1.shape[0] == pts2.shape[0]
    assert pts1.shape[0] >= 4
    if method == -1:
        assert pts1.shape[0] == 4
        H = cv2.getPerspectiveTransform(pts1, pts2)
    else:
        assert method in (0, cv2.RANSAC, cv2.LMEDS, cv2.RHO)
        H, mask = cv2.findHomography(srcPoints=pts1, dstPoints=pts2, method=method)
    return H


def estimate_F(pts1, pts2, method=cv2.FM_7POINT):
    assert pts1.shape[0] == pts2.shape[0]
    assert pts1.shape[0] >= 8
    F, mask = cv2.findFundamentalMat(points1=pts1, points2=pts2, method=method)
    return F, mask
