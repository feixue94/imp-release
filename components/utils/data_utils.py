import numpy as np


def norm_kpt(K, kp):
    kp = np.concatenate([kp, np.ones([kp.shape[0], 1])], axis=1)
    kp = np.matmul(kp, np.linalg.inv(K).T)[:, :2]
    return kp


def unnorm_kp(K, kp):
    kp = np.concatenate([kp, np.ones([kp.shape[0], 1])], axis=1)
    kp = np.matmul(kp, K.T)[:, :2]
    return kp


def interpolate_depth(pos, depth):
    # pos:[y,x]
    ids = np.array(range(0, pos.shape[0]))

    h, w = depth.shape

    i = pos[:, 0]
    j = pos[:, 1]
    valid_corner = np.logical_and(np.logical_and(i > 0, i < h - 1), np.logical_and(j > 0, j < w - 1))
    i, j = i[valid_corner], j[valid_corner]
    ids = ids[valid_corner]

    i_top_left = np.floor(i).astype(np.int32)
    j_top_left = np.floor(j).astype(np.int32)

    i_top_right = np.floor(i).astype(np.int32)
    j_top_right = np.ceil(j).astype(np.int32)

    i_bottom_left = np.ceil(i).astype(np.int32)
    j_bottom_left = np.floor(j).astype(np.int32)

    i_bottom_right = np.ceil(i).astype(np.int32)
    j_bottom_right = np.ceil(j).astype(np.int32)

    # Valid depth
    depth_top_left, depth_top_right, depth_down_left, depth_down_right = depth[i_top_left, j_top_left], depth[
        i_top_right, j_top_right], \
                                                                         depth[i_bottom_left, j_bottom_left], depth[
                                                                             i_bottom_right, j_bottom_right]

    valid_depth = np.logical_and(
        np.logical_and(
            depth_top_left > 0,
            depth_top_right > 0
        ),
        np.logical_and(
            depth_down_left > 0,
            depth_down_left > 0
        )
    )
    ids = ids[valid_depth]
    depth_top_left, depth_top_right, depth_down_left, depth_down_right = depth_top_left[valid_depth], depth_top_right[
        valid_depth], \
                                                                         depth_down_left[valid_depth], depth_down_right[
                                                                             valid_depth]

    i, j, i_top_left, j_top_left = i[valid_depth], j[valid_depth], i_top_left[valid_depth], j_top_left[valid_depth]

    # Interpolation
    dist_i_top_left = i - i_top_left.astype(np.float32)
    dist_j_top_left = j - j_top_left.astype(np.float32)
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
            w_top_left * depth_top_left +
            w_top_right * depth_top_right +
            w_bottom_left * depth_down_left +
            w_bottom_right * depth_down_right
    )
    return [interpolated_depth, ids]


def reprojection(depth_map, kpt, dR, dt, K1_img2depth, K1, K2):
    # warp kpt from img1 to img2
    def swap_axis(data):
        return np.stack([data[:, 1], data[:, 0]], axis=-1)

    kp_depth = unnorm_kp(K1_img2depth, kpt)
    uv_depth = swap_axis(kp_depth)
    z, valid_idx = interpolate_depth(uv_depth, depth_map)

    norm_kp = norm_kpt(K1, kpt)
    norm_kp_valid = np.concatenate([norm_kp[valid_idx, :], np.ones((len(valid_idx), 1))], axis=-1)
    xyz_valid = norm_kp_valid * z.reshape(-1, 1)
    xyz2 = np.matmul(xyz_valid, dR.T) + dt.reshape(1, 3)
    xy2 = xyz2[:, :2] / xyz2[:, 2:]
    kp2, valid = np.ones(kpt.shape) * 1e5, np.zeros(kpt.shape[0])
    kp2[valid_idx] = unnorm_kp(K2, xy2)
    valid[valid_idx] = 1
    return kp2, valid.astype(bool)


def reprojection_2s(kp1, kp2, depth1, depth2, K1, K2, dR, dt, size1, size2):
    # size:H*W
    depth_size1, depth_size2 = [depth1.shape[0], depth1.shape[1]], [depth2.shape[0], depth2.shape[1]]
    scale_1 = [float(depth_size1[0]) / size1[0], float(depth_size1[1]) / size1[1], 1]
    scale_2 = [float(depth_size2[0]) / size2[0], float(depth_size2[1]) / size2[1], 1]
    K1_img2depth, K2_img2depth = np.diag(np.asarray(scale_1)), np.diag(np.asarray(scale_2))
    kp1_2_proj, valid1_2 = reprojection(depth1, kp1, dR, dt, K1_img2depth, K1, K2)
    kp2_1_proj, valid2_1 = reprojection(depth2, kp2, dR.T, -np.matmul(dR.T, dt), K2_img2depth, K2, K1)
    return [kp1_2_proj, kp2_1_proj], [valid1_2, valid2_1]


def make_corr(kp1, kp2, desc1, desc2, depth1, depth2, K1, K2, dR, dt, size1, size2, corr_th, incorr_th,
              check_desc=False):
    # make reprojection
    [kp1_2, kp2_1], [valid1_2, valid2_1] = reprojection_2s(kp1, kp2, depth1, depth2, K1, K2, dR, dt, size1, size2)
    num_pts1, num_pts2 = kp1.shape[0], kp2.shape[0]
    # reprojection error
    dis_mat1 = np.sqrt(abs(
        (kp1 ** 2).sum(1, keepdims=True) + (kp2_1 ** 2).sum(1, keepdims=False)[np.newaxis] - 2 * np.matmul(kp1,
                                                                                                           kp2_1.T)))
    dis_mat2 = np.sqrt(abs(
        (kp2 ** 2).sum(1, keepdims=True) + (kp1_2 ** 2).sum(1, keepdims=False)[np.newaxis] - 2 * np.matmul(kp2,
                                                                                                           kp1_2.T)))
    repro_error = np.maximum(dis_mat1, dis_mat2.T)  # n1*n2

    # find corr index
    nn_sort1 = np.argmin(repro_error, axis=1)
    nn_sort2 = np.argmin(repro_error, axis=0)
    mask_mutual = nn_sort2[nn_sort1] == np.arange(kp1.shape[0])
    mask_inlier = np.take_along_axis(repro_error, indices=nn_sort1[:, np.newaxis], axis=-1).squeeze(1) < corr_th
    mask = mask_mutual & mask_inlier
    corr_index = np.stack([np.arange(num_pts1)[mask], np.arange(num_pts2)[nn_sort1[mask]]], axis=-1)

    if check_desc:
        # filter kpt in same pos using desc distance(e.g. DoG kpt)
        x1_valid, x2_valid = kp1[corr_index[:, 0]], kp2[corr_index[:, 1]]
        mask_samepos1 = np.logical_and(x1_valid[:, 0, np.newaxis] == kp1[np.newaxis, :, 0],
                                       x1_valid[:, 1, np.newaxis] == kp1[np.newaxis, :, 1])
        mask_samepos2 = np.logical_and(x2_valid[:, 0, np.newaxis] == kp2[np.newaxis, :, 0],
                                       x2_valid[:, 1, np.newaxis] == kp2[np.newaxis, :, 1])
        duplicated_mask = np.logical_or(mask_samepos1.sum(-1) > 1, mask_samepos2.sum(-1) > 1)
        duplicated_index = np.nonzero(duplicated_mask)[0]

        unique_corr_index = corr_index[~duplicated_mask]
        clean_duplicated_corr = []
        for index in duplicated_index:
            cur_desc1, cur_desc2 = desc1[mask_samepos1[index]], desc2[mask_samepos2[index]]
            cur_desc_mat = np.matmul(cur_desc1, cur_desc2.T)
            cur_max_index = [np.argmax(cur_desc_mat) // cur_desc_mat.shape[1],
                             np.argmax(cur_desc_mat) % cur_desc_mat.shape[1]]
            clean_duplicated_corr.append(np.stack([np.arange(num_pts1)[mask_samepos1[index]][cur_max_index[0]],
                                                   np.arange(num_pts2)[mask_samepos2[index]][cur_max_index[1]]]))

        clean_corr_index = unique_corr_index
        if len(clean_duplicated_corr) != 0:
            clean_duplicated_corr = np.stack(clean_duplicated_corr, axis=0)
            clean_corr_index = np.concatenate([clean_corr_index, clean_duplicated_corr], axis=0)
    else:
        clean_corr_index = corr_index
    # find incorr
    mask_incorr1 = np.min(dis_mat2.T[valid1_2], axis=-1) > incorr_th
    mask_incorr2 = np.min(dis_mat1.T[valid2_1], axis=-1) > incorr_th
    incorr_index1, incorr_index2 = np.arange(num_pts1)[valid1_2][mask_incorr1.squeeze()], \
                                   np.arange(num_pts2)[valid2_1][mask_incorr2.squeeze()]

    return clean_corr_index, incorr_index1, incorr_index2
