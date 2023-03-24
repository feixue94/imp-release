import numpy as np
import sys
import os
from copy import deepcopy

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from components.utils import evaluation_utils, metrics, fm_utils
import cv2


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


class auc_eval:
    def __init__(self, config):
        self.config = config
        self.err_r, self.err_t, self.err = [], [], []
        self.ms = []
        self.precision = []

    def run(self, info, th=1):
        E, r_gt, t_gt = info['e'], info['r_gt'], info['t_gt']
        K1, K2, img1, img2 = info['K1'], info['K2'], info['img1'], info['img2']
        corr_pts1, corr_pts2 = info['corr1'], info['corr2']
        corr1, corr2 = evaluation_utils.normalize_intrinsic(corr_pts1, K1), evaluation_utils.normalize_intrinsic(
            corr_pts2, K2)
        size1, size2 = max(img1.shape), max(img2.shape)
        scale1, scale2 = self.config['rescale'] / size1, self.config['rescale'] / size2
        # ransac
        ransac_th = 4. / ((K1[0, 0] + K1[1, 1]) * scale1 + (K2[0, 0] + K2[1, 1]) * scale2)
        # R_hat, t_hat, E_hat = self.estimate(corr1, corr2, ransac_th)
        R_hat, t_hat, E_hat = self.estimate_new(corr1=corr_pts1, corr2=corr_pts2, K1=K1, K2=K2, th=th)
        # get pose error
        err_r, err_t = metrics.evaluate_R_t(r_gt, t_gt, R_hat, t_hat)
        # print('err_r/t: ', err_r, err_t)
        # print('R/t: ', R_hat, t_hat)
        err = max(err_r, err_t)

        # print('corr1-2: ', corr1.shape, corr2.shape)
        if len(corr1) > 1:
            inlier_mask = metrics.compute_epi_inlier(corr1, corr2, E, self.config['inlier_th'])
            precision = inlier_mask.mean()
            ms = inlier_mask.sum() / len(info['x1'])
        else:
            ms = precision = 0

        return {'err_r': err_r, 'err_t': err_t, 'err': err, 'ms': ms, 'precision': precision}

    def res_inqueue(self, res):
        self.err_r.append(res['err_r']), self.err_t.append(res['err_t']), self.err.append(res['err'])
        self.ms.append(res['ms']), self.precision.append(res['precision'])

    def estimate(self, corr1, corr2, th):
        num_inlier = -1
        if corr1.shape[0] >= 5:
            E, mask_new = cv2.findEssentialMat(corr1, corr2, method=cv2.RANSAC, threshold=th, prob=1 - 1e-5)
            if E is None:  # or np.count_nonzero(~np.isnan(E)) > 0:
                E = [np.eye(3)]

            # print(E)
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _ = cv2.recoverPose(_E, corr1, corr2, np.eye(3), 1e9, mask=mask_new)
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    E = _E
        else:
            E, R, t = np.eye(3), np.eye(3), np.zeros(3)
        return R, t, E

    def estimate_new(self, corr1, corr2, K1, K2, th=1):
        if corr1.shape[0] >= 5:
            E, E_mask = cv2.findEssentialMat(points1=corr1,
                                             points2=corr2,
                                             cameraMatrix1=K1,
                                             cameraMatrix2=K2,
                                             distCoeffs1=None,
                                             distCoeffs2=None,
                                             threshold=th,
                                             prob=0.99999,
                                             mask=None,
                                             method=cv2.USAC_MAGSAC,
                                             # method=cv2.RANSAC,
                                             )
            if E is None or E.shape[0] != 3 or E.shape[1] != 3:
                E, R, t = np.eye(3), np.eye(3), np.zeros(3)
            else:
                R, t, mask_P = decompose_essestial_mat(E=E, pts0=corr1[E_mask.ravel() > 0],
                                                       pts1=corr2[E_mask.ravel() > 0],
                                                       K0=K1, K1=K2)

                mask = E_mask.ravel() >= 0
                mask[E_mask.ravel() > 0] = mask_P
        else:
            E, R, t = np.eye(3), np.eye(3), np.zeros(3)
        return R, t, E

    def parse(self):
        ths = np.arange(7) * 5
        approx_auc = metrics.approx_pose_auc(self.err, ths)
        exact_auc = metrics.pose_auc(self.err, ths)
        mean_pre, mean_ms = np.mean(np.asarray(self.precision)), np.mean(np.asarray(self.ms))

        # print('auc th: ', ths[1:])
        # print('approx auc: ', approx_auc)
        # print('exact auc: ', exact_auc)
        # print('mean match score: ', mean_ms * 100)
        # print('mean precision: ', mean_pre * 100)

        output = {
            'auc_th': ths[1:],
            'approx_auc': approx_auc,
            'exact_auc': exact_auc,
            'mean_match_score': mean_ms * 100,
            'mean_precision': mean_pre * 100,
        }

        return output


class FMbench_eval:

    def __init__(self, config):
        self.config = config
        self.pre, self.pre_post, self.sgd = [], [], []
        self.num_corr, self.num_corr_post = [], []

    def run(self, info, **kwargs):
        corr1, corr2 = info['corr1'], info['corr2']
        F = info['f']
        img1, img2 = info['img1'], info['img2']

        if len(corr1) > 1:
            pre_bf = fm_utils.compute_inlier_rate(corr1, corr2, np.flip(img1.shape[:2]), np.flip(img2.shape[:2]), F,
                                                  th=self.config['inlier_th']).mean()
            if len(corr1) >= 8:
                F_hat, mask_F = cv2.findFundamentalMat(corr1, corr2,
                                                       method=cv2.USAC_MAGSAC,
                                                       # method=cv2.FM_RANSAC,
                                                       ransacReprojThreshold=1,
                                                       confidence=1 - 1e-5)
            else:
                F_hat = None

            if F_hat is None:
                F_hat = np.ones([3, 3])
                mask_F = np.ones([len(corr1)]).astype(bool)
            else:
                mask_F = mask_F.squeeze().astype(bool)
            F_hat = F_hat[:3]
            pre_af = fm_utils.compute_inlier_rate(corr1[mask_F], corr2[mask_F], np.flip(img1.shape[:2]),
                                                  np.flip(img2.shape[:2]), F, th=self.config['inlier_th']).mean()
            num_corr_af = mask_F.sum()
            num_corr = len(corr1)
            sgd = fm_utils.compute_SGD(F, F_hat, np.flip(img1.shape[:2]), np.flip(img2.shape[:2]))
        else:
            pre_bf, pre_af, sgd = 0, 0, 1e8
            num_corr, num_corr_af = 0, 0
        return {'pre': pre_bf, 'pre_post': pre_af, 'sgd': sgd, 'num_corr': num_corr, 'num_corr_post': num_corr_af}

    def res_inqueue(self, res):
        self.pre.append(res['pre']), self.pre_post.append(res['pre_post']), self.sgd.append(res['sgd'])
        self.num_corr.append(res['num_corr']), self.num_corr_post.append(res['num_corr_post'])

    def parse(self):
        for seq_index in range(len(self.config['seq'])):
            seq = self.config['seq'][seq_index]
            offset = seq_index * 1000
            pre = np.asarray(self.pre)[offset:offset + 1000].mean()
            pre_post = np.asarray(self.pre_post)[offset:offset + 1000].mean()
            num_corr = np.asarray(self.num_corr)[offset:offset + 1000].mean()
            num_corr_post = np.asarray(self.num_corr_post)[offset:offset + 1000].mean()
            f_recall = (np.asarray(self.sgd)[offset:offset + 1000] < self.config['sgd_inlier_th']).mean()

            print(seq, 'results:')
            print('F_recall: ', f_recall)
            print('precision: ', pre)
            print('precision_post: ', pre_post)
            print('num_corr: ', num_corr)
            print('num_corr_post: ', num_corr_post, '\n')
