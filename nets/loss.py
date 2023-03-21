# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> loss
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   14/03/2022 17:27
=================================================='''
import torch
import torch.nn as nn


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class GraphLoss(nn.Module):
    default_config = {
        'match_threshold': 0.2,
        'multi_scale': False,
        'with_pose': False,
        'with_hard_negative': False,
        'neg_margin': 0.1,
        'with_sinkhorn': True,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.multi_scale = self.config['multi_scale']
        self.with_pose = self.config['with_pose']
        self.with_hard_negative = self.config['with_hard_negative']
        self.neg_margin = self.config['neg_margin']
        self.with_sinkhorn = self.config['with_sinkhorn']

    def forward(self, score, gt_matching_mask):
        output = {}
        batch_size = gt_matching_mask.shape[0]

        # m_loss_corr, m_loss_incorr, m_loss_neg = self.compute_matching_loss(pred_scores=score,
        #                                                                     gt_matching_mask=gt_matching_mask)
        match_loss_corr, match_loss_incorr, match_loss_neg = self.compute_matching_loss_batch(pred_scores=score,
                                                                                              gt_matching_mask=gt_matching_mask)

        match_loss = match_loss_corr + match_loss_incorr + match_loss_neg
        indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=score)

        # compute correct matches
        gt_matches = torch.max(gt_matching_mask[:, :-1, :], dim=-1, keepdim=False)[1]
        acc_corr = torch.sum(((indices0 - gt_matches) == 0) * (indices0 != -1) * (
                gt_matches < gt_matching_mask.shape[-1] - 1)) / batch_size
        acc_incorr = torch.sum((indices0 == -1) * (gt_matches == gt_matching_mask.shape[-1] - 1)) / batch_size
        acc_corr_total = torch.sum((gt_matches < gt_matching_mask.shape[-1] - 1)) / batch_size
        acc_incorr_total = torch.sum((gt_matches == gt_matching_mask.shape[-1] - 1)) / batch_size

        output['matching_loss'] = match_loss
        output['matching_loss_corr'] = match_loss_corr
        output['matching_loss_incorr'] = match_loss_incorr
        output['matching_loss_neg'] = match_loss_neg
        # output['pose_loss'] = torch.zeros_like(total_match_loss)

        # recover matches from the final output
        output['matches0'] = indices0  # use -1 for invalid match
        output['matches1'] = indices1  # use -1 for invalid match
        output['matching_scores0'] = mscores0
        output['matching_scores1'] = mscores1

        output['acc_corr'] = [acc_corr]
        output['acc_incorr'] = [acc_incorr]
        output['total_acc_corr'] = [acc_corr_total]
        output['total_acc_incorr'] = [acc_incorr_total]

        return output

    def compute_matching_loss_batch(self, pred_scores, gt_matching_mask):
        log_p = torch.log(abs(pred_scores) + 1e-8)

        num_corr = torch.sum(gt_matching_mask[:, :-1, :-1], dim=2).sum(dim=1)  # [B]
        num_corr[num_corr == 0] = 1
        loss_curr = torch.sum(log_p[:, :-1, :-1] * gt_matching_mask[:, :-1, :-1], dim=2).sum(dim=1)  # [B]
        loss_curr = loss_curr / num_corr
        loss_curr = -loss_curr.mean()

        num_incorr1 = torch.sum(gt_matching_mask[:, :, -1], dim=1)  # [B]
        num_incorr2 = torch.sum(gt_matching_mask[:, -1, :], dim=1)  # [B]
        loss_incorr1 = torch.sum(log_p[:, :, -1] * gt_matching_mask[:, :, -1], dim=1)
        loss_incorr2 = torch.sum(log_p[:, -1, :] * gt_matching_mask[:, -1, :], dim=1)

        incorr1_mask = (num_incorr1 > 0)
        incorr2_mask = (num_incorr2 > 0)

        if torch.sum(incorr1_mask) > 0:
            loss_incorr1 = loss_incorr1[incorr1_mask] / num_incorr1[incorr1_mask]
            loss_incorr2 = loss_incorr2[incorr2_mask] / num_incorr2[incorr2_mask]
            loss_incorr = -(loss_incorr1.mean() + loss_incorr2.mean()) / 2
        else:
            loss_incorr = torch.zeros(size=[], device=gt_matching_mask.device)

        if self.with_hard_negative:
            loss_neg = self.compute_matching_hard_negative_loss(pred_scores=pred_scores,
                                                                gt_matching_mask=gt_matching_mask)
        else:
            loss_neg = torch.zeros(size=[], device=gt_matching_mask.device)

        return loss_curr, loss_incorr, loss_neg

    def compute_matching_hard_negative_loss(self, pred_scores, gt_matching_mask):
        gt_matching_mask_inv = 1 - gt_matching_mask

        pos_row = torch.max(pred_scores[:, :-1, :] * gt_matching_mask[:, :-1, :], dim=2)[
            0]  # discard the last invalid cow
        pos_col = torch.max(pred_scores[:, :, :-1] * gt_matching_mask[:, :, :-1], dim=1)[
            0]  # discard the last invalid col
        neg_row = torch.max(pred_scores[:, :-1, :] * gt_matching_mask_inv[:, :-1, :], dim=2)[0]
        neg_col = torch.max(pred_scores[:, :, :-1] * gt_matching_mask_inv[:, :, :-1], dim=1)[0]

        # mask_row = ((pos_row - neg_row) < self.neg_margin)
        # mask_col = ((pos_col - neg_col) < self.neg_margin)
        loss_neg_row = -torch.clamp_max(pos_row - neg_row - self.neg_margin, max=0).mean()
        loss_neg_col = -torch.clamp_max(pos_col - neg_col - self.neg_margin, max=0).mean()

        loss_neg = (loss_neg_row + loss_neg_col) / 2.

        return loss_neg

    def compute_epipolar_loss_batch(self, pred_scores, epipolar_error):
        return (pred_scores * epipolar_error).mean()

    def compute_matching_loss(self, pred_scores, gt_matching_mask):
        log_p = torch.log(abs(pred_scores) + 1e-8)
        total_loss_corr, total_loss_incorr, total_loss_neg = 0, 0, 0

        gt_matching_mask_inv = 1 - gt_matching_mask

        min_prob = torch.min(log_p)

        batch_size = pred_scores.shape[0]
        n_valid_b = 0
        for bid in range(batch_size):
            cur_p = pred_scores[bid]
            cur_log_p = log_p[bid]
            num_corr = torch.sum(gt_matching_mask[bid, :-1, :-1])

            if num_corr > 0:
                loss_corr_pos = -torch.sum(cur_log_p[:-1, :-1] * gt_matching_mask[bid, :-1, :-1]) / num_corr
                total_loss_corr += loss_corr_pos

            num_incorr1 = torch.sum(gt_matching_mask[bid, :, -1])
            num_incorr2 = torch.sum(gt_matching_mask[bid, -1, :])

            if num_incorr1 == 0 or num_incorr2 == 0:
                total_loss_incorr += torch.zeros(size=[], device=gt_matching_mask.device)
            else:
                loss_incorr1 = -torch.sum(cur_log_p[:, -1] * gt_matching_mask[bid, :, -1]) / num_incorr1
                loss_incorr2 = -torch.sum(cur_log_p[-1, :] * gt_matching_mask[bid, -1, :]) / num_incorr2
                loss_incorr_pos = (loss_incorr1 + loss_incorr2) / 2
                n_valid_b += 1

                total_loss_incorr += loss_incorr_pos

            if self.with_hard_negative:
                pos_row = torch.max(cur_p[:-1, :] * gt_matching_mask[bid, :-1, :], dim=1)[
                    0]  # discard the last invalid cow
                pos_col = torch.max(cur_p[:, :-1] * gt_matching_mask[bid, :, :-1], dim=0)[
                    0]  # discard the last invalid col
                neg_row = torch.max(cur_p[:-1, :] * gt_matching_mask_inv[bid, :-1, :], dim=1)[0]
                neg_col = torch.max(cur_p[:, :-1] * gt_matching_mask_inv[bid, :, :-1], dim=0)[0]

                mask_row = ((pos_row - neg_row) < self.neg_margin)
                mask_col = ((pos_col - neg_col) < self.neg_margin)

                mask_valid = torch.zeros_like(cur_log_p)
                mask_valid[(gt_matching_mask[bid] > 0)] = min_prob - 1

                if torch.sum(mask_row) > 0:
                    a = cur_log_p[:-1, :] * gt_matching_mask_inv[bid, :-1, :] + mask_valid[:-1, :]
                    loss_neg_row = torch.max(a, dim=1)[0][mask_row]
                    loss_neg_row = loss_neg_row.mean()
                else:
                    loss_neg_row = torch.zeros(size=[], device=gt_matching_mask.device)

                if torch.sum(mask_col) > 0:
                    b = cur_log_p[:, :-1] * gt_matching_mask_inv[bid, :, :-1] + mask_valid[:, :-1]
                    loss_neg_col = torch.max(b, dim=0)[0][mask_col]
                    loss_neg_col = loss_neg_col.mean()
                else:
                    loss_neg_col = torch.zeros(size=[], device=gt_matching_mask.device)

                loss_neg = (loss_neg_row + loss_neg_col) / 2.

                total_loss_neg += loss_neg
            else:
                total_loss_neg = torch.zeros(size=[], device=gt_matching_mask.device)

        return total_loss_corr / batch_size, total_loss_incorr / n_valid_b, total_loss_neg / batch_size

    def compute_matches(self, scores):
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores0 = torch.where(mutual0, max0.values, zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return indices0, indices1, mscores0, mscores1
