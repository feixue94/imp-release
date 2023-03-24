import numpy as np


def line_to_border(line, size):
    # line:(a,b,c), ax+by+c=0
    # size:(W,H)
    H, W = size[1], size[0]
    a, b, c = line[0], line[1], line[2]
    epsa = 1e-8 if a >= 0 else -1e-8
    epsb = 1e-8 if b >= 0 else -1e-8
    intersection_list = []

    y_left = -c / (b + epsb)
    y_right = (-c - a * (W - 1)) / (b + epsb)
    x_top = -c / (a + epsa)
    x_down = (-c - b * (H - 1)) / (a + epsa)

    if y_left >= 0 and y_left <= H - 1:
        intersection_list.append([0, y_left])
    if y_right >= 0 and y_right <= H - 1:
        intersection_list.append([W - 1, y_right])
    if x_top >= 0 and x_top <= W - 1:
        intersection_list.append([x_top, 0])
    if x_down >= 0 and x_down <= W - 1:
        intersection_list.append([x_down, H - 1])
    if len(intersection_list) != 2:
        return None
    intersection_list = np.asarray(intersection_list)
    return intersection_list


def find_point_in_line(end_point):
    x_span, y_span = end_point[1, 0] - end_point[0, 0], end_point[1, 1] - end_point[0, 1]
    mv = np.random.uniform()
    point = np.asarray([end_point[0, 0] + x_span * mv, end_point[0, 1] + y_span * mv])
    return point


def epi_line(point, F):
    homo = np.concatenate([point, np.ones([len(point), 1])], axis=-1)
    epi = np.matmul(homo, F.T)
    return epi


def dis_point_to_line(line, point):
    homo = np.concatenate([point, np.ones([len(point), 1])], axis=-1)
    dis = line * homo
    dis = dis.sum(axis=-1) / (np.linalg.norm(line[:, :2], axis=-1) + 1e-8)
    return abs(dis)


def SGD_oneiter(F1, F2, size1, size2):
    H1, W1 = size1[1], size1[0]
    factor1 = 1 / np.linalg.norm(size1)
    factor2 = 1 / np.linalg.norm(size2)
    p0 = np.asarray([(W1 - 1) * np.random.uniform(), (H1 - 1) * np.random.uniform()])
    epi1 = epi_line(p0[np.newaxis], F1)[0]
    border_point1 = line_to_border(epi1, size2)
    if border_point1 is None:
        return -1

    p1 = find_point_in_line(border_point1)
    epi2 = epi_line(p0[np.newaxis], F2)
    d1 = dis_point_to_line(epi2, p1[np.newaxis])[0] * factor2
    epi3 = epi_line(p1[np.newaxis], F2.T)
    d2 = dis_point_to_line(epi3, p0[np.newaxis])[0] * factor1
    return (d1 + d2) / 2


def compute_SGD(F1, F2, size1, size2):
    np.random.seed(1234)
    N = 1000
    max_iter = N * 10
    count, sgd = 0, 0
    for i in range(max_iter):
        d1 = SGD_oneiter(F1, F2, size1, size2)
        if d1 < 0:
            continue
        d2 = SGD_oneiter(F2, F1, size1, size2)
        if d2 < 0:
            continue
        count += 1
        sgd += (d1 + d2) / 2
        if count == N:
            break
    if count == 0:
        return 1
    else:
        return sgd / count


def compute_inlier_rate(x1, x2, size1, size2, F_gt, th=0.003):
    t1, t2 = np.linalg.norm(size1) * th, np.linalg.norm(size2) * th
    epi1, epi2 = epi_line(x1, F_gt), epi_line(x2, F_gt.T)
    dis1, dis2 = dis_point_to_line(epi1, x2), dis_point_to_line(epi2, x1)
    mask_inlier = np.logical_and(dis1 < t2, dis2 < t1)
    return mask_inlier.mean() if len(mask_inlier) != 0 else np.array([0, 0])
