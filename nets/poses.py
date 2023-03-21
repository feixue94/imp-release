# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> poses
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   21/03/2022 14:35
=================================================='''
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from nets.utils_misc import _prob_sampling, _homo, _de_homo, _prob_sampling_simple, _prob_sampling_simple_v2, \
    _prob_sampling_simple_seq, _prob_sampling_simple_seq_v2
from nets.utils_misc import prob_sampling
from nets.utils_misc import _sampson_dist, _error_proj_H, _sampson_dist_general, _error_proj_H_general
import kornia.geometry as kG


# from pytorch3d.ops.perspective_n_points import efficient_pnp as epnp_torch

def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    F = F.reshape(-1, 1, 3, 3).repeat(1, num_pts, 1, 1)
    x2Fx1 = torch.matmul(x2.transpose(2, 3), torch.matmul(F, x1)).reshape(batch_size, num_pts)
    Fx1 = torch.matmul(F, x1).reshape(batch_size, num_pts, 3)
    Ftx2 = torch.matmul(F.transpose(2, 3), x2).reshape(batch_size, num_pts, 3)
    ys = x2Fx1 ** 2 * (
            1.0 / (Fx1[:, :, 0] ** 2 + Fx1[:, :, 1] ** 2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0] ** 2 + Ftx2[:, :, 1] ** 2 + 1e-15))
    return ys


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    # X = X.cpu()
    # X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        # e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        # upper = True
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, weights):
    # print('8pt: ', x_in.shape, weights.shape)
    # x_in: batch *  N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    # weights = weights / torch.sum(weights, dim=1, keepdim=True)  # need normalization?
    # x_in = x_in.squeeze(1)
    # print('weights:', weights)

    # Make input data (num_img_pair x num_corr x 4)
    # xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)
    xx = rearrange(x_in, 'b n c -> b c n')
    # xx = x_in

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    # print('X: ', X.shape)
    wX = torch.reshape(weights, (x_shp[0], x_shp[1], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class PoseEstimator(nn.Module):
    default_config = {
        'pose_type': 'H',
        'minium_samples': 3,
        'error_th': 8,
        'inlier_th': 0.25,
        'n_hypothesis': 128,
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**config, **self.default_config}
        self.pose_type = config['pose_type']
        self.minium_samples = config['minium_samples']
        self.n_hypothesis = config['n_hypothesis']
        self.error_th = config['error_th']
        self.inlier_th = config['inlier_th']

    def process_H(self, pts0, pts1, pred_matching_score, gt_matching_mask=None):
        '''
        :param matches: [N, M]
        :param pts0: [N, 2]
        :param pts1: [M, 2]
        :return:
        '''
        sampled_ids = _prob_sampling_simple_v2(matches=pred_matching_score, n_minium=self.minium_samples,
                                               n_hypothesis=self.n_hypothesis)  # [nH * nM, 2]
        pts0_i = pts0[sampled_ids[:, 0]].reshape(self.n_hypothesis, self.minium_samples, -1)
        pts1_i = pts1[sampled_ids[:, 1]].reshape(self.n_hypothesis, self.minium_samples, -1)
        H_i = kG.find_homography_dlt(points1=pts0_i, points2=pts1_i, weights=None)
        score = pred_matching_score[sampled_ids[:, 0], sampled_ids[:, 1]].reshape(self.n_hypothesis,
                                                                                  self.minium_samples)

        return H_i, score, sampled_ids.reshape(self.n_hypothesis, self.minium_samples, 2)

    def process_F(self, pts0, pts1, pred_matching_score, gt_matching_mask=None):
        '''
        :param matches: [N, M]
        :param pts0: [N, 2]
        :param pts1: [M, 2]
        :return:
        '''
        # sampled_ids = _prob_sampling(matches=matches, n_minium=self.minium_samples,
        #                              n_hypothesis=self.n_hypothesis)  # [nH, 2, nM]
        sampled_ids = _prob_sampling_simple_seq_v2(matches=pred_matching_score, n_minium=self.minium_samples,
                                                   n_hypothesis=self.n_hypothesis)  # [nH * nM, 2]
        # print('F sample_ids: ', sampled_ids.shape, pts0.shape, pts1.shape, matches.shape)

        score = pred_matching_score[sampled_ids[:, 0], sampled_ids[:, 1]].reshape(self.n_hypothesis,
                                                                                  self.minium_samples)

        pts0_i = pts0[sampled_ids[:, 0]].reshape(self.n_hypothesis, self.minium_samples, -1)
        pts1_i = pts1[sampled_ids[:, 1]].reshape(self.n_hypothesis, self.minium_samples, -1)
        weights = torch.ones_like(score)
        F_i = kG.find_fundamental(points1=pts0_i, points2=pts1_i, weights=weights)

        return F_i, score, sampled_ids.reshape(self.n_hypothesis, self.minium_samples, 2)

    def compute_inlier_ratio(self, P, pts0, pts1, error_th=5):
        if self.pose_type == 'H':
            error = _error_proj_H(H=P, X=pts0, Y=pts1)
        elif self.pose_type == 'F':
            error = _sampson_dist(F=P, X=pts0, Y=pts1)

        if len(pts0.shape) == 2:
            return torch.sum(error <= error_th) / pts0.shape[0]
        else:
            return torch.sum(error <= error_th) / (pts0.shape[0] * pts0.shape[1])

    def forward_eval(self, pts0, pts1, pred_scores, indices0):
        valid = (indices0 > -1)
        nG = torch.sum(valid)
        mpts0 = pts0[valid]
        mpts1 = pts1[indices0[valid]]
        if self.pose_type == 'H':
            pred_pose, sampled_scores, sampled_ids = self.process_H(pts0=pts0,
                                                                    pts1=pts1,
                                                                    pred_matching_score=pred_scores)
            nP = pred_pose.shape[0]
            proj_error_of_gt_matches = _error_proj_H(H=pred_pose,
                                                     X=mpts0[None].repeat(nP, 1, 1),
                                                     Y=mpts1[None].repeat(nP, 1, 1))
            proj_error_full = _error_proj_H_general(H=pred_pose, X=pts0[None].repeat(nP, 1, 1),
                                                    Y=pts1[None].repeat(nP, 1, 1))  # [nP, N, N]
        elif self.pose_type == 'F':
            pred_pose, sampled_scores, sampled_ids = self.process_F(pts0=pts0,
                                                                    pts1=pts1,
                                                                    pred_matching_score=pred_scores,
                                                                    )
            nP = pred_pose.shape[0]
            proj_error_of_gt_matches = _sampson_dist(F=pred_pose,
                                                     X=mpts0[None].repeat(nP, 1, 1),
                                                     Y=mpts1[None].repeat(nP, 1, 1))  # [nH, nG]
            proj_error_full = _sampson_dist_general(F=pred_pose, X=pts0[None].repeat(nP, 1, 1),
                                                    Y=pts1[None].repeat(nP, 1, 1))  # [nP, N, N]

        inlier_mask = (proj_error_of_gt_matches <= self.error_th)  # [nP, nG]
        inlier_ratio = torch.sum(inlier_mask, dim=1) / nG  # [nP]
        inlier_ratio[(inlier_ratio <= self.inlier_th)] = 0
        # print('proj_error_full: ', proj_error_full.shape, inlier_ratio.shape)

        support_connections = (proj_error_full <= self.error_th) * inlier_ratio[:, None, None].expand_as(
            proj_error_full)  # [nP, N, N]
        support_connections = torch.sum(support_connections, dim=0)  # [N, N]
        support_connections[support_connections > 0] = 1.
        output = {
            'pred_pose': pred_pose,
            'sampled_ids': sampled_ids,
            'inlier_ratio': inlier_ratio,
            'valid_pose': torch.sum((inlier_ratio >= self.inlier_th)),  # valid poses/batch
            'support_connections': support_connections,
        }

        return output

    def forward_train(self, pts0, pts1, pred_scores, gt_matching_mask):
        '''
        :param pts0: [B, N, 2]
        :param pts1: [B, M, 2]
        :param pred_scores: [B, N, M]
        :param gt_matching_mask: [B, N, M]
        :return:
        '''
        log_p = torch.log(abs(pred_scores) + 1e-8)
        bs = pts0.shape[0]

        all_poses = []
        all_inlier_ratios = []
        all_sampled_ids = []
        all_inlier_losss = []
        all_support_connections = []

        for bid in range(bs):
            gt_indices = torch.where(gt_matching_mask[bid] > 0)
            log_p_of_gt_matches = log_p[bid, gt_indices[0], gt_indices[1]]  # [nG]

            with torch.no_grad():
                # gt_indices = torch.where(gt_matching_mask[bid] > 0)
                nG = gt_indices[0].shape[0]
                if self.pose_type == 'H':
                    pred_pose, sampled_scores, sampled_ids = self.process_H(pts0=pts0[bid],
                                                                            pts1=pts1[bid],
                                                                            pred_matching_score=pred_scores[bid])
                    nP = pred_pose.shape[0]
                    proj_error_of_gt_matches = _error_proj_H(H=pred_pose,
                                                             X=pts0[bid:bid + 1, gt_indices[0]].repeat(nP, 1, 1),
                                                             Y=pts1[bid:bid + 1, gt_indices[1]].repeat(nP, 1, 1))
                    proj_error_full = _error_proj_H_general(H=pred_pose, X=pts0[bid].repeat(nP, 1, 1),
                                                            Y=pts1[bid].repeat(nP, 1, 1))  # [nP, N, N]
                elif self.pose_type == 'F':
                    pred_pose, sampled_scores, sampled_ids = self.process_F(pts0=pts0[bid],
                                                                            pts1=pts1[bid],
                                                                            pred_matching_score=pred_scores[bid],
                                                                            )
                    nP = pred_pose.shape[0]
                    proj_error_of_gt_matches = _sampson_dist(F=pred_pose,
                                                             X=pts0[bid:bid + 1, gt_indices[0]].repeat(nP, 1, 1),
                                                             Y=pts1[bid:bid + 1, gt_indices[1]].repeat(nP, 1,
                                                                                                       1))  # [nH, nG]
                    proj_error_full = _sampson_dist_general(F=pred_pose, X=pts0[bid].repeat(nP, 1, 1),
                                                            Y=pts1[bid].repeat(nP, 1, 1))  # [nP, N, N]

                inlier_mask = (proj_error_of_gt_matches <= self.error_th)  # [nP, nG]
                inlier_ratio = torch.sum(inlier_mask, dim=1) / nG  # [nP]
                inlier_ratio[(inlier_ratio <= self.inlier_th)] = 0
                # print('proj_error_full: ', proj_error_full.shape, inlier_ratio.shape)

                support_connections = (proj_error_full <= self.error_th) * inlier_ratio[:, None, None].expand_as(
                    proj_error_full)  # [nP, N, N]
                support_connections = torch.sum(support_connections, dim=0)  # [N, N]
                support_connections[support_connections > 0] = 1.

                all_support_connections.append(support_connections[None])

            inlier_loss = -log_p_of_gt_matches[None].repeat(nP, 1) * inlier_ratio[:, None].repeat(1, nG)
            inlier_loss = inlier_loss[inlier_mask].mean()

            all_poses.append(pred_pose[None])
            all_inlier_ratios.append(inlier_ratio[None])
            all_sampled_ids.append(sampled_ids[None])
            all_inlier_losss.append(inlier_loss[None])

        all_poses = torch.vstack(all_poses)  # [B, nH, 3, 3]
        # all_scores = torch.vstack(all_scores)  # [B, nG]
        all_inlier_ratios = torch.vstack(all_inlier_ratios)  # [B, nH]
        all_support_connections = torch.vstack(all_support_connections)  # [B, N, N]

        inlier_loss = torch.vstack(all_inlier_losss).mean()
        output = {
            'pred_pose': all_poses,
            'sampled_ids': torch.vstack(all_sampled_ids),
            'inlier_ratio': all_inlier_ratios,
            # 'pred_score': torch.mean(all_scores, dim=2),
            'pose_loss': inlier_loss.mean(),
            'valid_pose': torch.sum((all_inlier_ratios >= self.inlier_th)) / all_poses.shape[0],  # valid poses/batch
            'support_connections': all_support_connections,
        }

        return output

    def forward_train_v2(self, pts0, pts1, pred_scores, gt_matching_mask):
        '''
        :param pts0: [B, N, 2]
        :param pts1: [B, M, 2]
        :param pred_scores: [B, N, M]
        :param gt_matching_mask: [B, N, M]
        :return:
        '''
        log_p = torch.log(abs(pred_scores) + 1e-8)
        bs = pts0.shape[0]

        all_poses = []
        all_inlier_ratios = []
        all_sampled_ids = []
        all_inlier_losss = []
        for bid in range(bs):
            gt_indices = torch.where(gt_matching_mask[bid] > 0)
            log_p_of_gt_matches = log_p[bid, gt_indices[0], gt_indices[1]]  # [nG]

            with torch.no_grad():
                # gt_indices = torch.where(gt_matching_mask[bid] > 0)
                nG = gt_indices[0].shape[0]
                if self.pose_type == 'H':
                    pred_pose, sampled_scores, sampled_ids = self.process_H(pts0=pts0[bid],
                                                                            pts1=pts1[bid],
                                                                            pred_matching_score=pred_scores[bid])
                    nP = pred_pose.shape[0]
                    proj_error_of_gt_matches = _error_proj_H(H=pred_pose,
                                                             X=pts0[bid:bid + 1, gt_indices[0]].repeat(nP, 1, 1),
                                                             Y=pts1[bid:bid + 1, gt_indices[1]].repeat(nP, 1, 1))
                    # proj_error_full = _error_proj_H_general(H=pred_pose, X=pts0[bid].repeat(nP, 1, 1),
                    #                                         Y=pts1[bid].repeat(nP, 1, 1))  # [nP, N, N]
                elif self.pose_type == 'F':
                    pred_pose, sampled_scores, sampled_ids = self.process_F(pts0=pts0[bid],
                                                                            pts1=pts1[bid],
                                                                            pred_matching_score=pred_scores[bid],
                                                                            )
                    nP = pred_pose.shape[0]
                    proj_error_of_gt_matches = _sampson_dist(F=pred_pose,
                                                             X=pts0[bid:bid + 1, gt_indices[0]].repeat(nP, 1, 1),
                                                             Y=pts1[bid:bid + 1, gt_indices[1]].repeat(nP, 1,
                                                                                                       1))  # [nH, nG]

                inlier_mask = (proj_error_of_gt_matches <= self.error_th)  # [nP, nG]
                inlier_ratio = torch.sum(inlier_mask, dim=1) / nG  # [nP]
                inlier_ratio[(inlier_ratio <= self.inlier_th)] = 0

            inlier_loss = -log_p_of_gt_matches[None].repeat(nP, 1) * inlier_ratio[:, None].repeat(1, nG)
            inlier_loss = inlier_loss[inlier_mask].mean()

            all_poses.append(pred_pose[None])
            all_inlier_ratios.append(inlier_ratio[None])
            all_sampled_ids.append(sampled_ids[None])
            all_inlier_losss.append(inlier_loss[None])

        all_poses = torch.vstack(all_poses)  # [B, nH, 3, 3]
        # all_scores = torch.vstack(all_scores)  # [B, nG]
        all_inlier_ratios = torch.vstack(all_inlier_ratios)  # [B, nH]
        # all_support_connections = torch.vstack(all_support_connections)  # [B, N, N]
        inlier_loss = torch.vstack(all_inlier_losss).mean()
        # print('inlier_ratio: ', inlier_ratio, inlier_loss)

        output = {
            'pred_pose': all_poses,
            'sampled_ids': torch.vstack(all_sampled_ids),
            'inlier_ratio': all_inlier_ratios,
            # 'pred_score': torch.mean(all_scores, dim=2),
            'pose_loss': inlier_loss.mean(),
            'valid_pose': torch.sum((all_inlier_ratios >= self.inlier_th)) / all_poses.shape[0],  # valid poses/batch
            # 'support_connections': all_support_connections,
        }

        return output

    def forward_train_v3(self, pts0, pts1, pred_scores, gt_matching_mask):
        '''
        :param pts0: [B, N, 2]
        :param pts1: [B, M, 2]
        :param pred_scores: [B, N, M]
        :param gt_matching_mask: [B, N, M]
        :return:
        '''
        log_p = torch.log(abs(pred_scores) + 1e-8)
        bs = pts0.shape[0]

        all_poses = []
        all_inlier_ratios = []
        all_sampled_ids = []
        all_inlier_losss = []
        for bid in range(bs):
            gt_indices = torch.where(gt_matching_mask[bid] > 0)
            # log_p_of_gt_matches = log_p[bid, gt_indices[0], gt_indices[1]]  # [nG]

            batch_log_p = log_p[bid]  # [N, M]
            batch_gt_mask = gt_matching_mask[bid]

            with torch.no_grad():
                # gt_indices = torch.where(gt_matching_mask[bid] > 0)
                nG = gt_indices[0].shape[0]
                if self.pose_type == 'H':
                    pred_pose, sampled_scores, sampled_ids = self.process_H(pts0=pts0[bid],
                                                                            pts1=pts1[bid],
                                                                            pred_matching_score=pred_scores[bid])
                    nP = pred_pose.shape[0]
                    proj_error_of_gt_matches = _error_proj_H(H=pred_pose,
                                                             X=pts0[bid:bid + 1, gt_indices[0]].repeat(nP, 1, 1),
                                                             Y=pts1[bid:bid + 1, gt_indices[1]].repeat(nP, 1, 1))
                    # proj_error_full = _error_proj_H_general(H=pred_pose, X=pts0[bid].repeat(nP, 1, 1),
                    #                                         Y=pts1[bid].repeat(nP, 1, 1))  # [nP, N, N]
                elif self.pose_type == 'F':
                    pred_pose, sampled_scores, sampled_ids = self.process_F(pts0=pts0[bid],
                                                                            pts1=pts1[bid],
                                                                            pred_matching_score=pred_scores[bid],
                                                                            )
                    nP = pred_pose.shape[0]
                    proj_error_of_gt_matches = _sampson_dist(F=pred_pose,
                                                             X=pts0[bid:bid + 1, gt_indices[0]].repeat(nP, 1, 1),
                                                             Y=pts1[bid:bid + 1, gt_indices[1]].repeat(nP, 1,
                                                                                                       1))  # [nP, nG]

                inlier_mask = (proj_error_of_gt_matches <= self.error_th)  # [nP, nG]
                inlier_ratio = torch.sum(inlier_mask, dim=1) / nG  # [nP]
                inlier_ratio[(inlier_ratio < self.inlier_th)] = 0

            log_p_of_sampled_pts = batch_log_p[sampled_ids.view(-1, 2)[:, 0], sampled_ids.view(-1, 2)[:, 1]].view(nP,
                                                                                                                  self.minium_samples)
            gt_mask_of_sampled_pts = batch_gt_mask[sampled_ids.view(-1, 2)[:, 0], sampled_ids.view(-1, 2)[:, 1]].view(
                nP, self.minium_samples)

            inlier_loss = log_p_of_sampled_pts * gt_mask_of_sampled_pts * inlier_ratio[:, None].repeat(1,
                                                                                                       self.minium_samples)
            # inlier_loss = -log_p_of_gt_matches[None].repeat(nP, 1) * inlier_ratio[:, None].repeat(1, nG)
            # inlier_loss = inlier_loss[inlier_mask].mean()
            # print('inlier_loss: ', torch.max(inlier_ratio), torch.min(inlier_ratio))
            if torch.sum(torch.abs(inlier_loss) > 0) > 0:
                inlier_loss = inlier_loss[torch.abs(inlier_loss) > 0].mean()
            else:
                inlier_loss = torch.zeros(size=[], device=gt_matching_mask.device)

            inlier_loss = -inlier_loss

            all_poses.append(pred_pose[None])
            all_inlier_ratios.append(inlier_ratio[None])
            all_sampled_ids.append(sampled_ids[None])
            all_inlier_losss.append(inlier_loss[None])

        all_poses = torch.vstack(all_poses)  # [B, nH, 3, 3]
        # all_scores = torch.vstack(all_scores)  # [B, nG]
        all_inlier_ratios = torch.vstack(all_inlier_ratios)  # [B, nH]
        # all_support_connections = torch.vstack(all_support_connections)  # [B, N, N]
        inlier_loss = torch.vstack(all_inlier_losss).mean()
        # print('inlier_ratio: ', inlier_ratio, inlier_loss)

        output = {
            'pred_pose': all_poses,
            'sampled_ids': torch.vstack(all_sampled_ids),
            'inlier_ratio': all_inlier_ratios,
            # 'pred_score': torch.mean(all_scores, dim=2),
            'pose_loss': inlier_loss.mean(),
            'valid_pose': torch.sum((all_inlier_ratios >= self.inlier_th)) / all_poses.shape[0],  # valid poses/batch
            # 'support_connections': all_support_connections,
        }

        return output

    def forward(self, pts0, pts1, pred_scores, gt_matching_mask=None, indices0=None):
        if self.training:
            return self.forward_train_v3(pts0=pts0, pts1=pts1, pred_scores=pred_scores,
                                         gt_matching_mask=gt_matching_mask)
        else:
            return self.forward_eval(pts0=pts0, pts1=pts1, pred_scores=pred_scores, indices0=indices0)


class PoseEstimatorV2(nn.Module):
    default_config = {
        'pose_type': 'H',
        'minium_samples': 3,
        'error_th': 10,
        'inlier_th': 0.25,
        'n_hypothesis': 128,
        'n_connection_corss': 16,
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**config, **self.default_config}
        self.pose_type = config['pose_type']
        self.minium_samples = config['minium_samples']
        self.n_hypothesis = config['n_hypothesis']
        self.error_th = config['error_th']
        self.inlier_th = config['inlier_th']

    def process_F(self, pts0, pts1, pred_matching_score, **kwargs):
        '''
        :param matches: [N, M]
        :param pts0: [N, 2]
        :param pts1: [M, 2]
        :return:
        '''
        sampled_ids = _prob_sampling_simple_seq_v2(matches=pred_matching_score, n_minium=self.minium_samples,
                                                   n_hypothesis=self.n_hypothesis)  # [nH * nM, 2]
        score = pred_matching_score[sampled_ids[:, 0], sampled_ids[:, 1]].reshape(self.n_hypothesis,
                                                                                  self.minium_samples)

        pts0_i = pts0[sampled_ids[:, 0]].reshape(self.n_hypothesis, self.minium_samples, -1)
        pts1_i = pts1[sampled_ids[:, 1]].reshape(self.n_hypothesis, self.minium_samples, -1)
        weights = torch.ones_like(score)
        F_i = kG.find_fundamental(points1=pts0_i, points2=pts1_i, weights=weights)

        return F_i, score, sampled_ids.reshape(self.n_hypothesis, self.minium_samples, 2)

    def compute_inlier_ratio(self, P, pts0, pts1, error_th=5):
        if self.pose_type == 'H':
            error = _error_proj_H(H=P, X=pts0, Y=pts1)
        elif self.pose_type == 'F':
            error = _sampson_dist(F=P, X=pts0, Y=pts1)

        if len(pts0.shape) == 2:
            return torch.sum(error <= error_th) / pts0.shape[0]
        else:
            return torch.sum(error <= error_th) / (pts0.shape[0] * pts0.shape[1])

    def forward_eval(self, pts0, pts1, pred_scores, indices0):
        valid = (indices0 > -1)
        nG = torch.sum(valid)
        mpts0 = pts0[valid]
        mpts1 = pts1[indices0[valid]]
        pred_pose, sampled_scores, sampled_ids = self.process_F(pts0=pts0,
                                                                pts1=pts1,
                                                                pred_matching_score=pred_scores,
                                                                )
        nP = pred_pose.shape[0]
        proj_error_of_gt_matches = _sampson_dist(F=pred_pose,
                                                 X=mpts0[None].repeat(nP, 1, 1),
                                                 Y=mpts1[None].repeat(nP, 1, 1))  # [nH, nG]
        proj_error_full = _sampson_dist_general(F=pred_pose, X=pts0[None].repeat(nP, 1, 1),
                                                Y=pts1[None].repeat(nP, 1, 1))  # [nP, N, N]
        inlier_mask = (proj_error_of_gt_matches <= self.error_th)  # [nP, nG]
        inlier_ratio = torch.sum(inlier_mask, dim=1) / nG  # [nP]
        inlier_ratio[(inlier_ratio <= self.inlier_th)] = 0
        # print('proj_error_full: ', proj_error_full.shape, inlier_ratio.shape)

        # support_connections = (proj_error_full <= self.error_th) * inlier_ratio[:, None, None].expand_as(
        #     proj_error_full)  # [nP, N, N]
        # support_connections = torch.sum(support_connections, dim=0)  # [N, N]
        # support_connections[support_connections > 0] = 1.
        output = {
            'pred_pose': pred_pose,
            'sampled_ids': sampled_ids,
            'inlier_ratio': inlier_ratio,
            'valid_pose': torch.sum((inlier_ratio >= self.inlier_th)),  # valid poses/batch
            'proj_error_full': proj_error_full,
            # 'support_connections': support_connections,
        }

        return output

    def forward_train(self, pts0, pts1, pred_scores, gt_matching_mask):
        '''
        :param pts0: [B, N, 2]
        :param pts1: [B, M, 2]
        :param pred_scores: [B, N, M]
        :param gt_matching_mask: [B, N, M]
        :return:
        '''
        log_p = torch.log(abs(pred_scores) + 1e-8)
        bs = pts0.shape[0]

        all_poses = []
        all_inlier_ratios = []
        all_sampled_ids = []
        all_inlier_losss = []
        all_proj_err_full = []

        M = pred_scores.shape[1]
        N = pred_scores.shape[2]

        for bid in range(bs):
            gt_indices = torch.where(gt_matching_mask[bid] > 0)
            log_p_of_gt_matches = log_p[bid, gt_indices[0], gt_indices[1]]  # [nG]

            with torch.no_grad():
                # gt_indices = torch.where(gt_matching_mask[bid] > 0)
                nG = gt_indices[0].shape[0]
                pred_pose, sampled_scores, sampled_ids = self.process_F(pts0=pts0[bid],
                                                                        pts1=pts1[bid],
                                                                        pred_matching_score=pred_scores[bid],
                                                                        )
                nP = pred_pose.shape[0]
                proj_error_of_gt_matches = _sampson_dist(F=pred_pose,
                                                         X=pts0[bid:bid + 1, gt_indices[0]].repeat(nP, 1, 1),
                                                         Y=pts1[bid:bid + 1, gt_indices[1]].repeat(nP, 1,
                                                                                                   1))  # [nH, nG]
                proj_error_full = _sampson_dist_general(F=pred_pose, X=pts0[bid].repeat(nP, 1, 1),
                                                        Y=pts1[bid].repeat(nP, 1, 1))  # [nP, N, N]

                inlier_mask = (proj_error_of_gt_matches <= self.error_th)  # [nP, nG]
                inlier_ratio = torch.sum(inlier_mask, dim=1) / nG  # [nP]
                inlier_ratio[(inlier_ratio <= self.inlier_th)] = 0

            inlier_loss = -log_p_of_gt_matches[None].repeat(nP, 1) * inlier_ratio[:, None].repeat(1, nG)
            inlier_loss = inlier_loss[inlier_mask].mean()

            all_poses.append(pred_pose[None])
            all_inlier_ratios.append(inlier_ratio[None])
            all_sampled_ids.append(sampled_ids[None])
            all_inlier_losss.append(inlier_loss[None])
            all_proj_err_full.append(proj_error_full[None])

        all_poses = torch.vstack(all_poses)  # [B, nH, 3, 3]
        all_inlier_ratios = torch.vstack(all_inlier_ratios)  # [B, nH]
        all_proj_err_full = torch.vstack(all_proj_err_full)

        inlier_loss = torch.vstack(all_inlier_losss).mean()
        output = {
            'pred_pose': all_poses,
            'sampled_ids': torch.vstack(all_sampled_ids),
            'inlier_ratio': all_inlier_ratios,
            # 'pred_score': torch.mean(all_scores, dim=2),
            'pose_loss': inlier_loss.mean(),
            'valid_pose': torch.sum((all_inlier_ratios >= self.inlier_th)) / all_poses.shape[0],  # valid poses/batch
            'proj_error_full': all_proj_err_full,
        }

        return output

    def forward(self, pts0, pts1, pred_scores, gt_matching_mask=None, indices0=None):
        if self.training:
            return self.forward_train(pts0=pts0, pts1=pts1, pred_scores=pred_scores, gt_matching_mask=gt_matching_mask)
        else:
            return self.forward_eval(pts0=pts0, pts1=pts1, pred_scores=pred_scores, indices0=indices0)


class PoseEstimatorV3(nn.Module):
    default_config = {
        'pose_type': 'H',
        'minium_samples': 3,
        'error_th': 10,
        'inlier_th': 0.25,
        'n_hypothesis': 128,
        'n_connection_corss': 16,
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**config, **self.default_config}
        self.pose_type = config['pose_type']
        self.minium_samples = config['minium_samples']
        self.n_hypothesis = config['n_hypothesis']
        self.error_th = config['error_th']
        self.inlier_th = config['inlier_th']

    def forward_batch(self, pts0, pts1, indices, mscores0, mscores1, gt_pose, gt_matching_mask, fs=None):
        nB = indices.shape[0]
        total_pose_loss = torch.zeros(size=[], device=gt_matching_mask.device)
        total_geo_loss = torch.zeros(size=[], device=gt_matching_mask.device)

        n_pose = 0
        n_geo = 0
        for bid in range(nB):
            ids0 = torch.where(indices[bid] >= 0)[0]
            if ids0.shape[0] < 8:
                continue
            ids1 = indices[bid, ids0]
            sel_pts0 = pts0[bid, ids0][None]  # [B, N, 2]
            sel_pts1 = pts1[bid, ids1][None]  # [B, N, 2]
            sel_score0 = mscores0[bid, ids0][None]  # [B, N]
            sel_score1 = mscores1[bid, ids1][None]  # [B, N]
            sel_score = (sel_score0 + sel_score1) / 2

            # print(indices.shape, ids0.shape, ids1.shape, sel_pts0.shape, sel_pts1.shape, sel_score.shape)

            # pred_pose = kG.find_fundamental(points1=sel_pts0, points2=sel_pts1, weights=sel_score)
            # print('sel_pts0/1: ', sel_pts0.shape, sel_pts1.shape)
            pred_pose = weighted_8points(x_in=torch.cat([sel_pts0, sel_pts1], dim=2), weights=sel_score)
            gt_indices = torch.where(gt_matching_mask[bid] > 0)

            gt_pts0 = pts0[bid:bid + 1, gt_indices[0]]
            gt_pts1 = pts1[bid:bid + 1, gt_indices[1]]

            res_pred_pose = batch_episym(x1=gt_pts0, x2=gt_pts1, F=pred_pose)
            # res_pred_pose = batch_episym(x1=gt_pts0, x2=gt_pts1, F=gt_pose[bid:bid + 1])
            res_geo = batch_episym(x1=sel_pts0, x2=sel_pts1, F=gt_pose[bid:bid + 1])
            # print('pred_pose: ', pred_pose, gt_pose[bid])
            # res_gt = batch_episym(x1=gt_pts0, x2=gt_pts1, F=gt_pose[bid:bid + 1])
            # print('res_pred_pose: ', torch.min(res_pred_pose), torch.median(res_pred_pose), torch.max(res_pred_pose))
            # print('res_geo: ', torch.min(res_geo), torch.median(res_geo), torch.max(res_geo))

            res_pred_pose = torch.min(res_pred_pose, 5e-4 * res_pred_pose.new_ones(res_pred_pose.shape)) * 5e3
            res_geo = torch.min(res_geo, 5e-4 * res_geo.new_ones(res_geo.shape)) * 5e3

            total_pose_loss = total_pose_loss + res_pred_pose.mean()
            total_geo_loss = total_geo_loss + res_geo.mean()

            n_pose += 1
            n_geo += 1

        out = {
            'pose_loss': total_pose_loss / n_pose,
            'geo_loss': total_geo_loss / n_geo,
        }
        return out

    def forward(self, pts0, pts1, indices, mscores0, mscores1, gt_pose, gt_matching_mask, **kwargs):
        # print(pts0.shape, pts1.shape, indices.shape, mscores.shape, gt_pose.shape, gt_matching_mask.shape)
        fs = kwargs.get('fs') if 'fs' in kwargs.keys() else None

        if len(indices.shape) == 2:  # [B, N]
            return self.forward_batch(pts0=pts0, pts1=pts1, indices=indices,
                                      mscores0=mscores0,
                                      mscores1=mscores1,
                                      gt_pose=gt_pose,
                                      gt_matching_mask=gt_matching_mask, fs=fs)
        else:
            total_pose_loss = torch.zeros(size=[], device=gt_matching_mask.device)
            total_geo_loss = torch.zeros(size=[], device=gt_matching_mask.device)

            nI = indices.shape[1]  # [B, nI, N]
            for it in range(nI):
                out = self.forward_batch(pts0=pts0, pts1=pts1, indices=indices[:, it],
                                         mscores0=mscores0[:, it],
                                         mscores1=mscores1[:, it],
                                         gt_pose=gt_pose,
                                         gt_matching_mask=gt_matching_mask, fs=fs)

                total_pose_loss = total_pose_loss + out['pose_loss']
                total_geo_loss = total_geo_loss + out['geo_loss']

            out = {
                'pose_loss': total_pose_loss / nI,
                'geo_loss': total_geo_loss / nI,
            }
            return out


class UncertaintyEstimator(object):
    def __init__(self):
        pass

    def run(self, x0, x1, K0, K1, E, mscore):
        geo_unc = self.compute_geo_uncertainty(x0=x0, x1=x1, K0=K0, K1=K1, E=E)
        mat_unc = self.compute_matching_uncertainty(mscore=mscore)
        full_unc = geo_unc * mat_unc

        # a = np.mean(full_unc)
        # print('a: ', a, np.sqrt(np.var(full_unc)))
        # print('geo: ', np.min(geo_unc), np.median(geo_unc), np.max(geo_unc))
        # print('mat: ', np.min(mat_unc), np.median(mat_unc), np.max(mat_unc))
        return full_unc

    def compute_geo_uncertainty(self, x0, x1, K0, K1, E, max_dis=1e-3):
        # e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
        # E = E / np.linalg.norm(E, axis=1, keepdims=True)
        K = (K0 + K1) / 2.
        n0, n1 = x0.shape[0], x1.shape[0]
        x0_3d = (x0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        x1_3d = (x1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
        x0_h = np.concatenate([x0_3d, np.ones([n0, 1])], -1)
        x1_h = np.concatenate([x1_3d, np.ones([n1, 1])], -1)
        ep_line1 = E @ x0_h.T  # @ E.T
        ep_line2 = (x1_h @ E).T
        demo = ep_line1[0] ** 2 + ep_line1[1] ** 2 + ep_line2[0] ** 2 + ep_line2[1] ** 2
        dis = np.diag((x1_h @ (E @ x0_h.T))) ** 2 / demo

        # print('dis: ', dis.shape, np.min(dis), np.median(dis), np.max(dis))
        dis[dis > max_dis] = max_dis
        return np.exp(-dis)

    def compute_matching_uncertainty(self, mscore):
        '''
        :param mscore: np.array, (0, 1)
        :return:
        '''
        return mscore

    def run2(self, x0, x1, K0, K1, E, mscore0, mscore1, inlier_ratio, epi_th=2e-4, min_epi_th=1e-4, mscore_th=0.2):
        K = (K0 + K1) / 2.
        n0, n1 = x0.shape[0], x1.shape[0]
        x0_3d = (x0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        x1_3d = (x1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
        # x0_h = np.concatenate([x0_3d, np.ones([n0, 1])], -1)
        # x1_h = np.concatenate([x1_3d, np.ones([n1, 1])], -1)
        dist = _sampson_dist_general(F=torch.from_numpy(E).cuda(),
                                     X=torch.from_numpy(x0_3d).cpu(),
                                     Y=torch.from_numpy(x1_3d).cuda())
        min_dist0, _ = torch.min(dist, dim=1)
        min_dist1, _ = torch.min(dist, dim=0)
        valid_ids0 = (min_dist0 <= epi_th)
        valid_ids1 = (min_dist1 <= epi_th)

        gscore0 = torch.zeros_like(min_dist0)
        gscore0[valid_ids0] = 1.0

        gscore1 = torch.zeros_like(min_dist0)
        gscore1[valid_ids1] = 1.0

        score0 = gscore0 * mscore0 * inlier_ratio
        score1 = gscore1 * mscore1 * inlier_ratio

        return score0, score1
