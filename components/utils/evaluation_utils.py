import numpy as np
import h5py
import cv2


def normalize_intrinsic(x, K):
    # print(x,K)
    return (x - K[:2, 2]) / np.diag(K)[:2]


def normalize_size(x, size, scale=1):
    size = size.reshape([1, 2])
    norm_fac = size.max()
    norm_x = (x - size / 2 - 0.5) / (norm_fac * scale)
    return norm_x


def np_skew_symmetric(v):
    zero = np.zeros_like(v[:, 0])
    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)
    return M


def draw_points(img, points, color=(0, 255, 0), radius=3):
    dp = [(int(points[i, 0]), int(points[i, 1])) for i in range(points.shape[0])]
    for i in range(points.shape[0]):
        cv2.circle(img, dp[i], radius=radius, color=color)
    return img


def draw_match(img1, img2, corr1, corr2, inlier=[True], color=None, radius1=1, radius2=1, resize=None):
    if resize is not None:
        scale1, scale2 = [img1.shape[1] / resize[0], img1.shape[0] / resize[1]], [img2.shape[1] / resize[0],
                                                                                  img2.shape[0] / resize[1]]
        img1, img2 = cv2.resize(img1, resize, interpolation=cv2.INTER_AREA), cv2.resize(img2, resize,
                                                                                        interpolation=cv2.INTER_AREA)
        corr1, corr2 = corr1 / np.asarray(scale1)[np.newaxis], corr2 / np.asarray(scale2)[np.newaxis]
    corr1_key = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], radius1) for i in range(corr1.shape[0])]
    corr2_key = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], radius2) for i in range(corr2.shape[0])]

    assert len(corr1) == len(corr2)

    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]
    if color is None:
        color = [(0, 255, 0) if cur_inlier else (0, 0, 255) for cur_inlier in inlier]
    if len(color) == 1:
        display = cv2.drawMatches(img1, corr1_key, img2, corr2_key, draw_matches, None,
                                  matchColor=color[0],
                                  singlePointColor=color[0],
                                  flags=4
                                  )
    else:
        height, width = max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]
        display = np.zeros([height, width, 3], np.uint8)
        display[:img1.shape[0], :img1.shape[1]] = img1
        display[:img2.shape[0], img1.shape[1]:] = img2
        for i in range(len(corr1)):
            left_x, left_y, right_x, right_y = int(corr1[i][0]), int(corr1[i][1]), int(
                corr2[i][0] + img1.shape[1]), int(corr2[i][1])
            cur_color = (int(color[i][0]), int(color[i][1]), int(color[i][2]))
            cv2.line(display, (left_x, left_y), (right_x, right_y), cur_color, 1, lineType=cv2.LINE_AA)
    return display
