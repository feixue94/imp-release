# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> adgm
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   12/08/2022 15:28
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nets.gm import GM
from nets.gm import normalize_keypoints
from nets.layers import MultiHeadedAttention, MLP, SharedAttentionalPropagation
from nets.sampler import sample_ats, sample_score
from nets.basic_layers import sink_algorithm, dual_softmax
from nets.utils_misc import _sampson_dist_general


class AdaGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, pool_sizes: list = None, minium_nghs: int = 128,
                 sharing_layers: list = None,
                 with_packing=False, ac_fn='relu', norm_fn='bn', score_ratio=0.8):
        super().__init__()
        if sharing_layers is None:
            self.sharing_layers = [False for i in range(len(layer_names))]
        else:
            self.sharing_layers = sharing_layers
        self.layers = nn.ModuleList([
            SharedAttentionalPropagation(num_heads=4, feature_dim=feature_dim, sharing_attention=self.sharing_layers[i],
                                         ac_fn=ac_fn, norm_fn=norm_fn)
            for i in range(len(layer_names))
        ])
        self.names = layer_names
        self.pool_sizes = pool_sizes
        self.minium_nghs = minium_nghs
        self.with_packing = with_packing
        self.score_ratio = score_ratio

    def pooling_with_packing(self, X, last_packing, global_indexes, prob, pool_size, use_length_norm=True):
        '''
        :param X: [B, D, N] - updated X with the original size
        :param last_packing: [B, D, 1] - last packing node
        :param global_indexes: [B, K] - last indexes of nodes for message propagation
        :param prob: [B, K, N] (K <= N) - last attention matrix
        :param pool_size: int (>= 1) - pooling stride
        :return:
        '''
        assert pool_size > 0
        if pool_size == 1:
            Y = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
            return Y, global_indexes, last_packing

        if len(prob.shape) == 4:  # [B, H, K, N]
            sum_prob = torch.sum(prob, dim=1)  # [B, K, N]
            sum_prob = torch.sum(sum_prob, dim=1)  # [B, K]
        else:
            sum_prob = torch.sum(prob, dim=1)  # [B, K]
        if use_length_norm:  # apply the length of nodes to the score
            last_nodes = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)  # [B, D, K]
            last_nodes_length = F.normalize(last_nodes, dim=1, p=2)  # length of each node
            sum_prob = sum_prob * last_nodes_length
            norm_prob = sum_prob / torch.sum(sum_prob, dim=1, keepdim=True)  # normalized contribution
        else:
            norm_prob = sum_prob / torch.sum(sum_prob, dim=1, keepdim=True)  # normalized contribution

        nG = global_indexes.shape[1]  # number of the nodes for message propagation
        k = np.max([int(nG // pool_size), self.minium_nghs])
        k = np.min([k, nG])  # make sure the number of nodes >= minium_nghs

        values, indexes = torch.sort(norm_prob[:, :nG], dim=1, descending=True)  # :nG to avoid the packing node
        topk_indexes = indexes[:, :k]
        topk_indexes = torch.sort(topk_indexes, dim=1)[0]
        sel_indexes = torch.gather(global_indexes, index=topk_indexes, dim=1)
        Y = torch.gather(X, index=sel_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)

        nP = norm_prob.shape[1]
        if k == nP or not self.with_packing:
            return Y, sel_indexes, last_packing
        else:
            if nP == nG:  # no packing in the last iteration
                dis_indexes = indexes[:, k:]
                dis_scores = sum_prob[dis_indexes]  # use the raw sum score rather than normalized
                dis_indexes = torch.gather(global_indexes, index=dis_indexes, dim=1)
                dis_Y = torch.gather(X, index=dis_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)  # [B, C, N]
            else:
                dis_indexes = indexes[:, k:nG]
                dis_scores = sum_prob[dis_indexes]
                dis_indexes = torch.gather(global_indexes, index=dis_indexes, dim=1)
                dis_Y = torch.gather(X, index=dis_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)  # [B, C, N]

                # put last packing at the end
                dis_Y = torch.cat([dis_Y, last_packing], dim=-1)
                if use_length_norm:
                    dis_scores = torch.cat([dis_scores, sum_prob[:, nG:] * F.normalize(last_packing, dim=1, p=2)],
                                           dim=-1)
                else:
                    dis_scores = torch.cat([dis_scores, sum_prob[:, nG:]], dim=-1)

            new_packing = self.packing(x=dis_Y, score=dis_scores)
            return Y, sel_indexes, new_packing

    def ada_with_packing(self, X, last_packing, global_indexes, prob, use_length_norm=True):
        '''
        :param X: [B, D, N]
        :param last_packing: [B, D, 1]
        :param global_indices: [B, K]
        :param prob:[B, K, N]
        :return:
        '''

        if len(prob.shape) == 4:  # [B, H, K, N]
            sum_prob = torch.sum(prob, dim=1)  # [B, K, N]
            sum_prob = torch.sum(sum_prob, dim=1)  # [B, K]
        else:
            sum_prob = torch.sum(prob, dim=1)  # [B, K]
        if use_length_norm:  # apply the length of nodes to the score
            last_nodes = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)  # [B, D, K]
            last_nodes_length = F.normalize(last_nodes, dim=1, p=2)  # length of each node
            sum_prob = sum_prob * last_nodes_length
            norm_prob = sum_prob / torch.sum(sum_prob, dim=1, keepdim=True)  # normalized contribution
        else:
            norm_prob = sum_prob / torch.sum(sum_prob, dim=1, keepdim=True)  # normalized contribution

        nG = global_indexes.shape[1]  # number of the nodes for message propagation
        if nG <= self.minium_nghs:
            Y = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
            return Y, global_indexes, None

        sel_indexes, dis_indexes = sample_ats(X=norm_prob, K=None)
        Y = torch.gather(X, index=sel_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)

        k = sel_indexes.shape[1]
        nP = norm_prob.shape[1]
        if k == nP or not self.with_packing:
            return Y, sel_indexes, None
        else:
            if nP == nG:
                dis_scores = sum_prob[dis_indexes]
                dis_indexes = torch.gather(global_indexes, index=dis_indexes, dim=1)
                dis_Y = torch.gather(X, index=dis_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)  # [B, C, N]
            else:
                dis_scores = sum_prob[dis_indexes]
                dis_indexes = torch.gather(global_indexes, index=dis_indexes, dim=1)
                dis_Y = torch.gather(X, index=dis_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)  # [B, C, N]

                # put last packing at the end
                dis_Y = torch.cat([dis_Y, last_packing], dim=-1)
                if use_length_norm:
                    dis_scores = torch.cat([dis_scores, sum_prob[:, nG:] * F.normalize(last_packing, dim=1, p=2)],
                                           dim=-1)
                else:
                    dis_scores = torch.cat([dis_scores, sum_prob[:, nG:]], dim=-1)
            new_packing = self.packing(x=dis_Y, score=dis_scores)
            return Y, sel_indexes, new_packing

    def packing(self, x, score):
        '''
        :param x:
        :param score:
        :return:
        '''
        norm_score = score / torch.sum(score, dim=1, keepdim=True)
        # print('norm: ', norm_score.shape)
        y = x * norm_score[:, None, :].expand_as(x)
        y = torch.sum(y, dim=2, keepdim=True)  # [B, C, 1]
        return y

    def ada_pooling(self, X, M, prob, pool_size=1, **kwargs):
        '''
        :param X: [B, C, N]
        :param M: [B, N, K]
        :param prob: [B, N, N] or [B, H, N, K]
        :return: newM[B, N, K]
        '''
        assert pool_size > 0
        if pool_size == 1:  # no updating global indexes
            return M

        bs = X.shape[0]
        if len(prob.shape) == 4:  # [B, H, N, K]
            sumProb = torch.sum(prob, dim=1)
            sumProb = torch.sum(sumProb, dim=1) * torch.norm(X,
                                                             dim=1)  # [B, K], discarded tokens already masked with 0, no worry
        else:
            sumProb = torch.sum(prob, dim=1) * torch.norm(X, dim=1)
        normProb = sumProb / torch.sum(sumProb, dim=1, keepdim=True)  # [B, K]
        sumM = torch.mean(M, dim=1)  # [B, K]

        # if torch.sum(sumM, dim=1).shape[1] <= self.minium_nghs:  # dont update M
        #     return M

        bids, mids = torch.where(sumM > 0)
        newM = torch.zeros_like(sumM)
        for b in range(bs):
            gids = mids[bids == b]
            scores = normProb[b][gids]
            if gids.shape[0] <= self.minium_nghs:
                sel_gids = gids
            else:
                if self.score_ratio > 0:
                    sel_ids, dis_ids = sample_score(X=scores[None], ratio=self.score_ratio)
                else:
                    sel_ids, dis_ids = sample_ats(X=scores[None], K=None)
                sel_gids = gids[sel_ids]
            newM[b][sel_gids] = 1.

            # print(b, gids.shape, sel_ids.shape, sel_gids.shape)

        newM = newM.unsqueeze(1).repeat(1, X.shape[2], 1)
        return newM

    def ada_pooling_index(self, X, global_indexes, prob, pool_size=1, **kwargs):
        '''
        :param X: [B, D, N]
        :param global_indexes: [B, K]
        :param prob: [B, N, K] or [B, H, N, K]
        :return:
        '''
        # print('ada_pooling_by_index: ', X.shape, global_indexes.shape, prob.shape, pool_size)
        assert pool_size > 0
        assert global_indexes.shape[1] == prob.shape[-1]
        if pool_size == 1 or global_indexes.shape[1] <= self.minium_nghs:
            Y = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
            return Y, global_indexes

        bs = X.shape[0]
        if len(prob.shape) == 4:  # [B, H, N, K]
            sumProb = torch.sum(prob, dim=1)
            Y = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
            sumProb = torch.sum(sumProb, dim=1) * torch.norm(Y, dim=1)
        else:
            Y = torch.gather(X, index=global_indexes[:, None, :].repeat(1, X.shape[1], 1), dim=2)
            sumProb = torch.sum(prob, dim=1) * torch.norm(Y, dim=1)
        normProb = sumProb / torch.sum(sumProb, dim=1, keepdim=True)

        # print('prob: ', sumProb.shape, X.shape, global_indexes.shape, prob.shape)

        new_global_indexes = []
        Y = []
        for b in range(bs):
            scores = normProb[b]
            # print('scores: ', scores.shape)
            if self.score_ratio > 0:
                sel_ids, dis_ids = sample_score(X=scores[None], ratio=self.score_ratio)
            else:
                sel_ids, dis_ids = sample_ats(X=scores[None], K=None)
            sel_gids = global_indexes[b][sel_ids]
            # print('sel_gids: ', sel_gids.shape, sel_ids.shape, dis_ids.shape)
            new_global_indexes.append(sel_gids[None])
            # a = X[b, :, sel_gids][None]
            # print('a: ', a.shape)
            Y.append(X[b, :, sel_gids][None])

        if len(Y) == 1:
            return Y[0], new_global_indexes[0]
        else:  # should not work unless all samples have the same number of sel_gids
            return torch.cat(Y, dim=0), torch.cat(new_global_indexes, dim=0)

    def forward(self, desc0, desc1):
        if self.training:
            return self.forward_train(desc0=desc0, desc1=desc1)
        else:
            return self.forward_test(desc0=desc0, desc1=desc1)

    def forward_train(self, desc0, desc1):
        print('forward_train desc: ', desc0.shape, desc1.shape)
        # for iteration
        all_desc0s = []
        all_desc1s = []

        # for shared attention
        self_attentions = [None]
        cross_attentions = [None]

        bs = desc0.shape[0]
        N0 = desc0.shape[2]
        N1 = desc1.shape[2]
        dev = desc0.device

        M00 = torch.ones(N0, N0, device=dev)[None].repeat(bs, 1, 1).float()
        M11 = torch.ones(N1, N1, device=dev)[None].repeat(bs, 1, 1).float()
        M01 = torch.ones(N0, N1, device=dev)[None].repeat(bs, 1, 1).float()
        M10 = torch.ones(N1, N0, device=dev)[None].repeat(bs, 1, 1).float()

        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                if self.pool_sizes[i] > 0:  # 1 for old M, 2 for updated M
                    M = self.ada_pooling(X=torch.cat([desc0, desc1], dim=0),
                                         M=torch.cat([M01, M10], dim=0),
                                         prob=cross_prob, pool_size=self.pool_sizes[i])
                    M01 = M[:bs]
                    M10 = M[bs:]

                crossM = torch.cat([M10, M01], dim=0)
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc1, desc0], dim=0),
                              cross_attentions[-1], M=crossM)
                cross_prob = torch.cat([layer.prob[bs:], layer.prob[:bs]], dim=0)  # [1->0, 0->1]->[0->1,1->0]
                cross_attentions.append(layer.prob)

                delta0 = delta[:bs]
                delta1 = delta[bs:]

            elif name == 'self':
                if self.pool_sizes[i] > 0:  # 1 for old M, 2 for updated M
                    M = self.ada_pooling(X=torch.cat([desc0, desc1], dim=0),
                                         M=torch.cat([M00, M11], dim=0),
                                         prob=self_prob, pool_size=self.pool_sizes[i])
                    M00 = M[:bs]
                    M11 = M[bs:]

                selfM = torch.cat([M00, M11], dim=0)
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc0, desc1], dim=0),
                              self_attentions[-1], M=selfM)
                self_prob = layer.prob
                self_attentions.append(layer.prob)

                delta0 = delta[:bs]
                delta1 = delta[bs:]

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

            # print(i, name, self.pool_sizes[i],
            #       torch.mean(M00, dim=1).sum(dim=1).mean(),
            #       torch.mean(M11, dim=1).sum(dim=1).mean(),
            #       torch.mean(M01, dim=1).sum(dim=1).mean(),
            #       torch.mean(M10, dim=1).sum(dim=1).mean())

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)

        return all_desc0s, all_desc1s, []

    def forward_test(self, desc0, desc1):
        # for iteration
        all_desc0s = []
        all_desc1s = []

        # for shared attention
        self_attentions00 = [None]
        self_attentions11 = [None]
        cross_attentions01 = [None]
        cross_attentions10 = [None]

        bs = desc0.shape[0]
        N0 = desc0.shape[2]
        N1 = desc1.shape[2]
        dev = desc0.device

        self_indexes00 = torch.arange(0, N0, device=dev)[None]
        self_indexes11 = torch.arange(0, N1, device=dev)[None]
        cross_indexes01 = torch.arange(0, N0, device=dev)[None]
        cross_indexes10 = torch.arange(0, N1, device=dev)[None]

        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':  # 1 for sharing indexes, 2 for updating indexes
                if self.pool_sizes[i] > 0:
                    ds_desc0, cross_indexes01 = self.ada_pooling_index(X=desc0, global_indexes=cross_indexes01,
                                                                       prob=cross_attentions01[-1],
                                                                       pool_size=self.pool_sizes[i])
                    ds_desc1, cross_indexes10 = self.ada_pooling_index(X=desc1, global_indexes=cross_indexes10,
                                                                       prob=cross_attentions10[-1],
                                                                       pool_size=self.pool_sizes[i])
                else:
                    ds_desc0 = desc0
                    ds_desc1 = desc1

                delta0 = layer(desc0, ds_desc1, cross_attentions10[-1], M=None)
                cross_attentions10.append(layer.prob)
                delta1 = layer(desc1, ds_desc0, cross_attentions01[-1], M=None)
                cross_attentions01.append(layer.prob)

            elif name == 'self':
                if self.pool_sizes[i] > 0:  # 1 for sharing indexes, 2 for updating indexes
                    ds_desc0, self_indexes00 = self.ada_pooling_index(X=desc0, global_indexes=self_indexes00,
                                                                      prob=self_attentions00[-1],
                                                                      pool_size=self.pool_sizes[i])
                    ds_desc1, self_indexes11 = self.ada_pooling_index(X=desc1, global_indexes=self_indexes11,
                                                                      prob=self_attentions11[-1],
                                                                      pool_size=self.pool_sizes[i])
                else:
                    ds_desc0 = desc0
                    ds_desc1 = desc1

                delta0 = layer(desc0, ds_desc0, self_attentions00[-1], M=None)
                self_attentions00.append(layer.prob)
                delta1 = layer(desc1, ds_desc1, self_attentions11[-1], M=None)
                self_attentions11.append(layer.prob)

            # print(i, name, self.pool_sizes[i], ds_desc0.shape, ds_desc1.shape)

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)
            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)

        return all_desc0s, all_desc1s, []


class AdaGMN(GM):
    def __init__(self, config={}):
        # pool_sizes = [0, 0] * 2 + [2, 2, 1, 1] * 21
        self.pool_sizes = [0, 0] * 2 + [0, 0, 0, 0] * 21
        self.sharing_layers = [False, False] * 2 + [False, False, True, True] * 21
        super().__init__(config={**config, **{'pool_sizes': self.pool_sizes}})

        self.gnn = AdaGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pool_sizes=self.pool_sizes,
            with_packing=False,
            ac_fn=self.config['ac_fn'],
            sharing_layers=self.sharing_layers,
            norm_fn=self.config['norm_fn'],
        )

        self.n_min_tokens = self.config['n_min_tokens']

        self.with_ada = True
        self.first_it_to_update = 2

        self.self_indexes0 = None
        self.self_indexes1 = None
        self.self_prob0 = None
        self.self_prob1 = None

        self.cross_indexes0 = None
        self.cross_indexes1 = None
        self.cross_prob0 = None
        self.cross_prob1 = None

        self.last_self_packs0 = None
        self.last_self_packs1 = None
        self.last_cross_packs0 = None
        self.last_cross_packs1 = None

    def forward_train(self, data, p=0.2):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        nK0 = desc0.shape[-1]
        nK1 = desc1.shape[-1]
        nI = self.config['n_layers']
        nB = desc0.shape[0]

        dev = desc0.device
        dust_id = torch.Tensor([desc0.shape[-1]]).to(dev)
        global_ids0 = torch.arange(0, desc0.shape[-1], device=dev, requires_grad=False)
        global_ids1 = torch.arange(0, desc1.shape[-1], device=dev, requires_grad=False)
        all_gids0 = [global_ids0 for i in range(nB)]
        all_gids1 = [global_ids1 for i in range(nB)]

        total_loss_corr = torch.zeros(size=[], device=dev)
        total_loss_incorr = torch.zeros(size=[], device=dev)
        total_loss_neg = torch.zeros(size=[], device=dev)

        M00 = None
        M01 = None
        M10 = None
        M11 = None
        prob00 = None
        prob01 = None
        prob11 = None
        prob10 = None

        all_indices0 = []
        all_indices1 = []
        all_mscores0 = []
        all_mscores1 = []

        # gids0 = global_ids0
        # gids1 = global_ids1

        for ni in range(nI):
            # print('\n.....................Process ni: ', ni, desc0[0, 0, gids0][:5], desc1[0, 0, gids1][:5],
            #       gids0.shape,
            #       gids1.shape)
            # self attention first
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0,
                           prob=None if prob00 is None else prob00,
                           M=None if M00 is None else M00)
            prob00 = layer.prob
            delta1 = layer(desc1, desc1,
                           prob=None if prob11 is None else prob11,
                           M=None if M11 is None else M11)
            prob11 = layer.prob
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1
            # print('self desc0: ', ni, desc0[0, 0, gids0][0:5])
            # print('self desc1: ', ni, desc1[0, 0, gids1][0:5])

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1,
                           prob=None if prob10 is None else prob10,
                           M=None if M10 is None else M10)
            prob10 = layer.prob
            delta1 = layer(desc1, desc0,
                           prob=None if prob01 is None else prob01,
                           M=None if M01 is None else M01)
            prob01 = layer.prob
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # print('cross desc0: ', ni, desc0[0, 0, gids0][0:5])
            # print('cross desc1: ', ni, desc1[0, 0, gids1][0:5])

            if not self.multi_proj:
                mdesc0 = self.final_proj(desc0)
                mdesc1 = self.final_proj(desc1)
            else:
                mdesc0 = self.final_proj[ni](desc0)
                mdesc1 = self.final_proj[ni](desc1)

            if ni < self.first_it_to_update:
                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                loss = self.match_net.compute_matching_loss_batch(pred_scores=pred_score,
                                                                  gt_matching_mask=data['matching_mask'])
                total_loss_corr = total_loss_corr + loss[0]
                total_loss_incorr = total_loss_incorr + loss[1]
                total_loss_neg = total_loss_neg + loss[2]

                indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score)
                all_indices0.append(indices0)
                all_indices1.append(indices1)
                all_mscores0.append(mscores0)
                all_mscores1.append(mscores1)
                # all_pred_scores.append(pred_score)

            else:
                batch_loss_corr = torch.zeros(size=[], device=dev)
                batch_loss_incorr = torch.zeros(size=[], device=dev)
                batch_loss_neg = torch.zeros(size=[], device=dev)
                batch_indices0 = torch.zeros((nB, nK0), device=dev, requires_grad=False).long() - 1
                batch_indices1 = torch.zeros((nB, nK1), device=dev, requires_grad=False).long() - 1
                batch_mscores0 = torch.zeros((nB, nK0), device=dev, requires_grad=False)
                batch_mscores1 = torch.zeros((nB, nK1), device=dev, requires_grad=False)

                # if not self.sharing_layers[ni * 2]:
                perform_updating = self.sharing_layers[2 * ni]

                if perform_updating:
                    with torch.no_grad():
                        sum_prob00 = torch.sum(prob00, dim=1).sum(dim=1)
                        sum_prob01 = torch.sum(prob01, dim=1).sum(dim=1)
                        sum_prob10 = torch.sum(prob10, dim=1).sum(dim=1)
                        sum_prob11 = torch.sum(prob11, dim=1).sum(dim=1)

                        norm_prob00 = sum_prob00 / torch.sum(sum_prob00, dim=1, keepdim=True)
                        norm_prob01 = sum_prob01 / torch.sum(sum_prob01, dim=1, keepdim=True)
                        norm_prob10 = sum_prob10 / torch.sum(sum_prob10, dim=1, keepdim=True)
                        norm_prob11 = sum_prob11 / torch.sum(sum_prob11, dim=1, keepdim=True)

                        M00 = torch.zeros((nB, desc0.shape[-1], desc0.shape[-1]), device=dev, requires_grad=False)
                        M01 = torch.zeros((nB, desc1.shape[-1], desc0.shape[-1]), device=dev,
                                          requires_grad=False)  # [target, source]
                        M11 = torch.zeros((nB, desc1.shape[-1], desc1.shape[-1]), device=dev, requires_grad=False)
                        M10 = torch.zeros((nB, desc0.shape[-1], desc1.shape[-1]), device=dev, requires_grad=False)

                for bi in range(nB):  # used for next iteration
                    gids0 = all_gids0[bi]
                    gids1 = all_gids1[bi]

                    sel_mdesc0 = mdesc0[bi, :, gids0][None]
                    sel_mdesc1 = mdesc1[bi, :, gids1][None]

                    dist = torch.einsum('bdn,bdm->bnm', sel_mdesc0, sel_mdesc1)
                    dist = dist / self.config['descriptor_dim'] ** .5
                    # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                    pred_score = self.compute_score(dist=dist, dustbin=self.bin_score,
                                                    iteration=self.sinkhorn_iterations)

                    # print(ni, pred_score.shape, pred_score)
                    indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=0.2)
                    indices0 = indices0[0]
                    indices1 = indices1[0]
                    mscores0 = mscores0[0]
                    mscores1 = mscores1[0]
                    valid0 = (indices0 >= 0)
                    valid1 = (indices1 >= 0)
                    # print(gids0[valid0].shape, gids1[indices0[valid0]].shape)
                    batch_indices0[bi, gids0[valid0]] = gids1[indices0[valid0]]
                    batch_indices1[bi, gids1[valid1]] = gids0[indices1[valid1]]
                    batch_mscores0[bi, gids0] = mscores0
                    batch_mscores1[bi, gids1] = mscores1

                    ns0 = sel_mdesc0.shape[-1]
                    ns1 = sel_mdesc1.shape[-1]
                    index0 = torch.hstack([gids0, dust_id])[:, None].repeat(1, ns1 + 1).reshape(-1, 1).long()
                    index1 = torch.hstack([gids1, dust_id])[None, :].repeat(ns0 + 1, 1).reshape(-1,
                                                                                                1).long()  # [(n0+1) x (n1+1), 2]

                    gt_score = data['matching_mask'][bi][index0, index1].reshape(ns0 + 1, ns1 + 1)[None]
                    # avoid some matched keypoints do not exist
                    gt_score[:, :-1, ns1] = 1 - torch.max(gt_score[:, :-1, :-1], dim=2)[0]
                    gt_score[:, ns0, :-1] = 1 - torch.max(gt_score[:, :-1, :-1], dim=1)[0]

                    loss = self.match_net.compute_matching_loss_batch(pred_scores=pred_score, gt_matching_mask=gt_score)
                    batch_loss_corr = batch_loss_corr + loss[0]
                    batch_loss_incorr = batch_loss_incorr + loss[1]
                    batch_loss_neg = batch_loss_neg + loss[2]

                    if perform_updating:  # when not sharing attention, update!
                        with torch.no_grad():
                            if self.n_min_tokens > 0 and gids0.shape[-1] <= self.n_min_tokens:
                                update_0 = False
                            else:
                                update_0 = True

                            if self.n_min_tokens > 0 and gids1.shape[-1] <= self.n_min_tokens:
                                update_1 = False
                            else:
                                update_1 = True

                            if update_0:
                                _, pids0 = torch.where(torch.sum(pred_score[:, :-1, :-1], dim=-1) >= 0.1)

                                md_prob00 = torch.median(norm_prob00[bi][gids0][pids0])
                                md_prob01 = torch.median(norm_prob01[bi][gids0][pids0])
                                aug_ids00 = torch.where(norm_prob00[bi][gids0] >= md_prob00)[0]
                                aug_ids01 = torch.where(norm_prob01[bi][gids0] >= md_prob01)[0]

                                full_ids0 = torch.unique(torch.hstack([pids0, aug_ids00, aug_ids01]))
                                gids0 = gids0[full_ids0]

                            if update_1:
                                _, pids1 = torch.where(torch.sum(pred_score[:, :-1, :-1], dim=1) >= 0.1)

                                md_prob10 = torch.median(norm_prob10[bi][gids1][pids1])
                                md_prob11 = torch.median(norm_prob11[bi][gids1][pids1])
                                aug_ids10 = torch.where(norm_prob10[bi][gids1] >= md_prob10)[0]
                                aug_ids11 = torch.where(norm_prob11[bi][gids1] >= md_prob11)[0]

                                full_ids1 = torch.unique(torch.hstack([pids1, aug_ids10, aug_ids11]))
                                gids1 = gids1[full_ids1]

                            all_gids0[bi] = gids0
                            all_gids1[bi] = gids1

                            # update M here
                            ng0 = gids0.shape[-1]
                            ng1 = gids1.shape[-1]

                            # M00[bi,
                            #     gids0[:, None].repeat(1, ng0).reshape(-1).long(),
                            #     gids0[None, :].repeat(ng0, 1).reshape(-1).long()] = 1
                            # M11[bi,
                            #     gids1[:, None].repeat(1, ng1).reshape(-1).long(),
                            #     gids1[None, :].repeat(ng1, 1).reshape(-1).long()] = 1
                            # M01[bi,
                            #     gids1[:, None].repeat(1, ng0).reshape(-1).long(),
                            #     gids0[None, :].repeat(ng1, 1).reshape(-1).long()] = 1
                            # M10[bi,
                            #     gids0[:, None].repeat(1, ng1).reshape(-1).long(),
                            #     gids1[None, :].repeat(ng0, 1).reshape(-1).long()] = 1
                            M00[bi, :, gids0.long()] = 1
                            M01[bi, :, gids0.long()] = 1
                            M11[bi, :, gids1.long()] = 1
                            M10[bi, :, gids1.long()] = 1

                            # print('Updating: ', ni, update_0, update_1, gids0.shape, gids1.shape)

                total_loss_corr = total_loss_corr + batch_loss_corr / nB
                total_loss_incorr = total_loss_incorr + batch_loss_incorr / nB
                total_loss_neg = total_loss_neg + batch_loss_neg / nB

                all_indices0.append(batch_indices0)
                all_indices1.append(batch_indices1)
                all_mscores0.append(batch_mscores0)
                all_mscores1.append(batch_mscores1)

            # print(ni, torch.sum(all_indices0[-1] >= 0))

        total_loss_corr = total_loss_corr / nI
        total_loss_incorr = total_loss_incorr / nI
        total_loss_neg = total_loss_neg / nI
        total_loss_match = total_loss_corr + total_loss_incorr + total_loss_neg

        with torch.no_grad():
            gt_matching_mask = data['matching_mask'].repeat(nI, 1, 1)
            gt_matches = torch.max(gt_matching_mask[:, :-1, :], dim=-1, keepdim=False)[1]
            acc_corr = torch.sum(
                ((torch.cat(all_indices0, dim=0) - gt_matches) == 0) * (torch.cat(all_indices0, dim=0) != -1) * (
                        gt_matches < gt_matching_mask.shape[-1] - 1)) / gt_matches.shape[0]
            acc_incorr = torch.sum(
                (torch.cat(all_indices0, dim=0) == -1) * (gt_matches == gt_matching_mask.shape[-1] - 1)) / \
                         gt_matches.shape[
                             0]
            acc_corr_total = torch.sum((gt_matches < gt_matching_mask.shape[-1] - 1)) / gt_matches.shape[0]
            acc_incorr_total = torch.sum((gt_matches == gt_matching_mask.shape[-1] - 1)) / gt_matches.shape[0]

        return {
            'scores': [pred_score],
            'loss': total_loss_match,
            'matching_loss': total_loss_match,
            'matching_loss_corr': total_loss_corr,
            'matching_loss_incorr': total_loss_incorr,
            'matching_loss_neg': total_loss_neg,
            'matches0': all_indices0,
            'matches1': all_indices1,
            'matching_scores0': all_mscores0,
            'matching_scores1': all_mscores1,

            'acc_corr': [acc_corr],
            'acc_incorr': [acc_incorr],
            'total_acc_corr': [acc_corr_total],
            'total_acc_incorr': [acc_incorr_total],

            'indices0': all_indices0,
            'mscores0': all_mscores0,
        }

    def forward_train_with_pose_v2(self, data):
        match_out = self.forward_train(data=data)
        all_indices0 = match_out['matches0']
        all_mscores0 = match_out['matching_scores0']
        all_mscores1 = match_out['matching_scores1']
        nI = self.config['n_layers']

        e_gt = torch.reshape(data['gt_E'], (-1, 9))
        K0s = data['intrinsics0']
        K1s = data['intrinsics1']
        pose_out = self.pose_net(
            data['keypoints0_3d'].repeat(nI, 1, 1),
            data['keypoints1_3d'].repeat(nI, 1, 1),
            # kpts0.repeat(nI, 1, 1),
            # kpts1.repeat(nI, 1, 1),
            torch.cat(all_indices0, dim=0),
            torch.cat(all_mscores0, dim=0),
            torch.cat(all_mscores1, dim=0),
            e_gt.repeat(nI, 1),
            data['matching_mask'][:, :-1, :-1].repeat(nI, 1, 1),
            fs=(K0s[:, 0, 0] + K0s[:, 1, 1] + K1s[:, 0, 0] + K1s[:, 1, 1]).repeat(nI, 1) / 4,
        )

        pose_loss = pose_out['pose_loss']
        geo_loss = pose_out['geo_loss']
        total_loss_pose = pose_loss + geo_loss

        match_out['loss'] = match_out['loss'] + total_loss_pose
        match_out['pose_loss'] = total_loss_pose

        return match_out

    def produce_matches(self, data, p=0.2, mscore_th=0.1, uncertainty_ratio=1., **kwargs):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        scores0, scores1 = data['scores0'], data['scores1']

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        nK0 = desc0.shape[-1]
        nK1 = desc1.shape[-1]
        nI = self.config['n_layers']
        nB = desc0.shape[0]

        dev = desc0.device
        global_ids0 = torch.arange(0, desc0.shape[-1], device=dev, requires_grad=False)
        global_ids1 = torch.arange(0, desc1.shape[-1], device=dev, requires_grad=False)
        all_gids0 = [global_ids0 for i in range(nB)]
        all_gids1 = [global_ids1 for i in range(nB)]
        gids0 = global_ids0
        gids1 = global_ids1

        M00 = None
        M01 = None
        M10 = None
        M11 = None
        prob00 = None
        prob01 = None
        prob11 = None
        prob10 = None

        all_indices0 = []
        all_indices1 = []
        all_mscores0 = []
        all_mscores1 = []

        for ni in range(nI):
            # print('\n.....................Process ni: ', ni, desc0[0, 0, gids0][:5], desc1[0, 0, gids1][:5],
            #       gids0.shape,
            #       gids1.shape)

            # self attention first
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0,
                           prob=None if prob00 is None else prob00,
                           M=None if M00 is None else M00)
            prob00 = layer.prob
            delta1 = layer(desc1, desc1,
                           prob=None if prob11 is None else prob11,
                           M=None if M11 is None else M11)
            prob11 = layer.prob
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1
            # print('self desc0: ', ni, desc0[0, 0, gids0][0:5])
            # print('self desc1: ', ni, desc1[0, 0, gids1][0:5])

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1,
                           prob=None if prob10 is None else prob10,
                           M=None if M10 is None or ni == 3 else M10)
            prob10 = layer.prob
            delta1 = layer(desc1, desc0,
                           prob=None if prob01 is None else prob01,
                           M=None if M01 is None or ni == 3 else M01)
            prob01 = layer.prob
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # print('cross desc0: ', ni, desc0[0, 0, gids0][0:5])
            # print('cross desc1: ', ni, desc1[0, 0, gids1][0:5])

            if not self.multi_proj:
                mdesc0 = self.final_proj(desc0)
                mdesc1 = self.final_proj(desc1)
            else:
                mdesc0 = self.final_proj[ni](desc0)
                mdesc1 = self.final_proj[ni](desc1)

            if ni < self.first_it_to_update:
                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score,
                                                                              p=p)
                all_indices0.append(indices0)
                # all_indices1.append(indices1)
                all_mscores0.append(mscores0)
                # all_mscores1.append(mscores1)
            else:
                batch_indices0 = torch.zeros((nB, nK0), device=dev, requires_grad=False).long() - 1
                # batch_indices1 = torch.zeros((nB, nK1), device=dev, requires_grad=False).long() - 1
                batch_mscores0 = torch.zeros((nB, nK0), device=dev, requires_grad=False)
                # batch_mscores1 = torch.zeros((nB, nK1), device=dev, requires_grad=False)

                perform_updating = self.sharing_layers[2 * ni]
                if perform_updating:
                    sum_prob00 = torch.sum(prob00, dim=1).sum(dim=1)
                    sum_prob01 = torch.sum(prob01, dim=1).sum(dim=1)
                    sum_prob10 = torch.sum(prob10, dim=1).sum(dim=1)
                    sum_prob11 = torch.sum(prob11, dim=1).sum(dim=1)

                    norm_prob00 = sum_prob00 / torch.sum(sum_prob00, dim=1, keepdim=True)
                    norm_prob01 = sum_prob01 / torch.sum(sum_prob01, dim=1, keepdim=True)
                    norm_prob10 = sum_prob10 / torch.sum(sum_prob10, dim=1, keepdim=True)
                    norm_prob11 = sum_prob11 / torch.sum(sum_prob11, dim=1, keepdim=True)

                    # print('sum_p00: ', sum_prob00[sum_prob00 > 0][:5], sum_prob00[sum_prob00 > 0].shape)
                    # print('sum_p01: ', sum_prob01[sum_prob01 > 0][:5], sum_prob01[sum_prob01 > 0].shape)
                    # print('sum_p11: ', sum_prob11[sum_prob11 > 0][:5], sum_prob11[sum_prob11 > 0].shape)
                    # print('sum_p10: ', sum_prob10[sum_prob10 > 0][:5], sum_prob10[sum_prob10 > 0].shape)

                    M00 = torch.zeros((nB, desc0.shape[-1], desc0.shape[-1]), device=dev, requires_grad=False)
                    M01 = torch.zeros((nB, desc1.shape[-1], desc0.shape[-1]), device=dev,
                                      requires_grad=False)  # [target, source]
                    M11 = torch.zeros((nB, desc1.shape[-1], desc1.shape[-1]), device=dev, requires_grad=False)
                    M10 = torch.zeros((nB, desc0.shape[-1], desc1.shape[-1]), device=dev, requires_grad=False)

                for bi in range(nB):  # used for next iteration
                    gids0 = all_gids0[bi]
                    gids1 = all_gids1[bi]
                    sel_mdesc0 = mdesc0[bi, :, gids0][None]
                    sel_mdesc1 = mdesc1[bi, :, gids1][None]

                    dist = torch.einsum('bdn,bdm->bnm', sel_mdesc0, sel_mdesc1)
                    dist = dist / self.config['descriptor_dim'] ** .5
                    # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                    pred_score = self.compute_score(dist=dist, dustbin=self.bin_score,
                                                    iteration=self.sinkhorn_iterations)

                    indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
                    indices0 = indices0[0]
                    indices1 = indices1[0]
                    mscores0 = mscores0[0]
                    mscores1 = mscores1[0]
                    valid0 = (indices0 >= 0)
                    valid1 = (indices1 >= 0)
                    batch_indices0[bi, gids0[valid0]] = gids1[indices0[valid0]]
                    # batch_indices1[bi, gids1[valid1]] = gids0[indices1[valid1]]
                    batch_mscores0[bi, gids0] = mscores0
                    # batch_mscores1[bi, gids1] = mscores1

                    if perform_updating:  # when not sharing attention, update!
                        with torch.no_grad():
                            if self.n_min_tokens > 0 and gids0.shape[-1] <= self.n_min_tokens:
                                update_0 = False
                            else:
                                update_0 = True

                            if self.n_min_tokens > 0 and gids1.shape[-1] <= self.n_min_tokens:
                                update_1 = False
                            else:
                                update_1 = True

                            if update_0:
                                _, pids0 = torch.where(
                                    torch.sum(pred_score[:, :-1, :-1], dim=-1) >= mscore_th * uncertainty_ratio)
                                # print('pids0: ', ni, pids0)
                                if pids0.shape[-1] > 0:
                                    md_prob00 = torch.median(norm_prob00[bi][gids0][pids0])
                                    md_prob01 = torch.median(norm_prob01[bi][gids0][pids0])
                                    aug_ids00 = torch.where(norm_prob00[bi][gids0] >= md_prob00)[0]
                                    aug_ids01 = torch.where(norm_prob01[bi][gids0] >= md_prob01)[0]
                                    full_ids0 = torch.unique(torch.hstack([pids0, aug_ids00, aug_ids01]))
                                    gids0 = gids0[full_ids0]

                            if update_1:
                                _, pids1 = torch.where(
                                    torch.sum(pred_score[:, :-1, :-1], dim=1) >= mscore_th * uncertainty_ratio)
                                if pids1.shape[-1] > 0:
                                    md_prob10 = torch.median(norm_prob10[bi][gids1][pids1])
                                    md_prob11 = torch.median(norm_prob11[bi][gids1][pids1])
                                    aug_ids10 = torch.where(norm_prob10[bi][gids1] >= md_prob10)[0]
                                    aug_ids11 = torch.where(norm_prob11[bi][gids1] >= md_prob11)[0]

                                    full_ids1 = torch.unique(torch.hstack([pids1, aug_ids10, aug_ids11]))
                                    gids1 = gids1[full_ids1]

                            all_gids0[bi] = gids0
                            all_gids1[bi] = gids1

                            ### update M here
                            M00[bi, :, gids0.long()] = 1
                            M01[bi, :, gids0.long()] = 1
                            M11[bi, :, gids1.long()] = 1
                            M10[bi, :, gids1.long()] = 1

                            # print('Updating: ', update_0, update_1, gids0.shape, gids1.shape)
                all_indices0.append(batch_indices0)
                all_mscores0.append(batch_mscores0)

        acc_corr = torch.zeros(size=[], device=desc0.device) + 0
        acc_corr_total = torch.zeros(size=[], device=desc0.device) + 1
        acc_incorr = torch.zeros(size=[], device=desc0.device) + 0
        acc_incorr_total = torch.zeros(size=[], device=desc0.device) + 1
        output = {
            'scores': [pred_score],
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'acc_corr': [acc_corr],
            'acc_incorr': [acc_incorr],
            'total_acc_corr': [acc_corr_total],
            'total_acc_incorr': [acc_incorr_total],
        }

        return output

    def produce_matches_test(self, data, p=0.2, mscore_th=0.1, uncertainty_ratio=1., **kwargs):
        # print('In produce_matches_test....')
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        scores0, scores1 = data['scores0'], data['scores1']

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        nK0 = desc0.shape[-1]
        nK1 = desc1.shape[-1]
        nI = self.config['n_layers']
        nB = desc0.shape[0]

        assert nB == 1

        dev = desc0.device
        global_ids0 = torch.arange(0, desc0.shape[-1], device=dev, requires_grad=False)
        global_ids1 = torch.arange(0, desc1.shape[-1], device=dev, requires_grad=False)
        full_ids0 = global_ids0
        full_ids1 = global_ids1
        sample_ids0 = []
        sample_ids1 = []

        M00 = None
        M01 = None
        M10 = None
        M11 = None
        prob00 = None
        prob01 = None
        prob11 = None
        prob10 = None

        all_indices0 = []
        all_mscores0 = []

        for ni in range(nI):
            # for ni in range(5):
            # print('\n....................Process ni: ', ni, desc0[0, 0, 0:5], desc1[0, 0, 0:5], desc0.shape,
            #       desc1.shape)
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0,
                           prob=None if prob00 is None else prob00,
                           M=None if M00 is None else M00)
            prob00 = layer.prob
            delta1 = layer(desc1, desc1,
                           prob=None if prob11 is None else prob11,
                           M=None if M11 is None else M11)
            prob11 = layer.prob
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # print('self desc0: ', ni, desc0[:, 0, 0:5])
            # print('self desc1: ', ni, desc1[:, 0, 0:5])

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1,
                           prob=None if prob10 is None else prob10,
                           M=None if M10 is None or ni == 3 else M10)
            prob10 = layer.prob
            delta1 = layer(desc1, desc0,
                           prob=None if prob01 is None else prob01,
                           M=None if M01 is None or ni == 3 else M01)
            prob01 = layer.prob
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # print('cross desc0: ', ni, desc0[:, 0, 0:5])
            # print('cross desc1: ', ni, desc1[:, 0, 0:5])

            if not self.multi_proj:
                mdesc0 = self.final_proj(desc0)
                mdesc1 = self.final_proj(desc1)
            else:
                mdesc0 = self.final_proj[ni](desc0)
                mdesc1 = self.final_proj[ni](desc1)

            if ni < self.first_it_to_update:
                # if ni < 3:
                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
                all_indices0.append(indices0)
                # all_indices1.append(indices1)
                all_mscores0.append(mscores0)
                # all_mscores1.append(mscores1)

                # keep all samples
                sample_ids0.append(global_ids0)
                sample_ids1.append(global_ids1)

            else:  # pooling based on the results of 1
                bi = 0
                full_indices0 = torch.zeros((nB, nK0), device=dev, requires_grad=False).long() - 1
                full_mscores0 = torch.zeros((nB, nK0), device=dev, requires_grad=False)

                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
                indices0 = indices0[bi]
                mscores0 = mscores0[bi]
                valid0 = (indices0 >= 0)

                full_indices0[bi, sample_ids0[-1][valid0]] = sample_ids1[-1][indices0[valid0]]
                full_mscores0[bi, sample_ids0[-1]] = mscores0

                all_indices0.append(full_indices0)
                all_mscores0.append(full_mscores0)

                perform_updating = self.sharing_layers[2 * ni]
                if not perform_updating:
                    sample_ids0.append(sample_ids0[-1])
                    sample_ids1.append(sample_ids1[-1])
                else:
                    sum_prob00 = torch.sum(prob00, dim=1).sum(dim=1)
                    sum_prob01 = torch.sum(prob01, dim=1).sum(dim=1)
                    sum_prob10 = torch.sum(prob10, dim=1).sum(dim=1)
                    sum_prob11 = torch.sum(prob11, dim=1).sum(dim=1)

                    norm_prob00 = sum_prob00 / torch.sum(sum_prob00, dim=1, keepdim=True)
                    norm_prob01 = sum_prob01 / torch.sum(sum_prob01, dim=1, keepdim=True)
                    norm_prob10 = sum_prob10 / torch.sum(sum_prob10, dim=1, keepdim=True)
                    norm_prob11 = sum_prob11 / torch.sum(sum_prob11, dim=1, keepdim=True)

                    # print('sum_p00: ', sum_prob00[0, :5], sum_prob00.shape)
                    # print('sum_p01: ', sum_prob01[0, :5], sum_prob01.shape)
                    # print('sum_p11: ', sum_prob11[0, :5], sum_prob11.shape)
                    # print('sum_p10: ', sum_prob10[0, :5], sum_prob10.shape)
                    with torch.no_grad():
                        if self.n_min_tokens > 0 and sample_ids0[-1].shape[-1] <= self.n_min_tokens:
                            update_0 = False
                        else:
                            update_0 = True

                        if self.n_min_tokens > 0 and sample_ids1[-1].shape[-1] <= self.n_min_tokens:
                            update_1 = False
                        else:
                            update_1 = True

                        if update_0:
                            _, pids0 = torch.where(
                                torch.sum(pred_score[:, :-1, :-1], dim=-1) >= mscore_th * uncertainty_ratio)
                            # print('pids0: ', ni, pids0)

                            if pids0.shape[-1] > 0:
                                md_prob00 = torch.median(norm_prob00[bi][pids0])
                                md_prob01 = torch.median(norm_prob01[bi][pids0])
                                aug_ids00 = torch.where(norm_prob00[bi] >= md_prob00)[0]
                                aug_ids01 = torch.where(norm_prob01[bi] >= md_prob01)[0]
                                full_ids0 = torch.unique(torch.hstack([pids0, aug_ids00, aug_ids01]))

                                sample_ids0.append(sample_ids0[-1][full_ids0])
                        else:
                            sample_ids0.append(sample_ids0[-1])

                        if update_1:
                            _, pids1 = torch.where(
                                torch.sum(pred_score[:, :-1, :-1], dim=1) >= mscore_th * uncertainty_ratio)
                            if pids1.shape[-1] > 0:
                                md_prob10 = torch.median(norm_prob10[bi][pids1])
                                md_prob11 = torch.median(norm_prob11[bi][pids1])
                                aug_ids10 = torch.where(norm_prob10[bi] >= md_prob10)[0]
                                aug_ids11 = torch.where(norm_prob11[bi] >= md_prob11)[0]
                                full_ids1 = torch.unique(torch.hstack([pids1, aug_ids10, aug_ids11]))

                                # print('full_ids1: ', desc1[bi, :].shape, full_ids1.shape)
                                sample_ids1.append(sample_ids1[-1][full_ids1])
                        else:
                            sample_ids1.append(sample_ids1[-1])

                        # update probability
                        if update_0 and update_1:
                            desc0 = desc0[:, :, full_ids0]
                            desc1 = desc1[:, :, full_ids1]

                        elif update_0 and (not update_1):
                            desc0 = desc0[:, :, full_ids0]
                        elif (not update_0) and update_1:
                            desc1 = desc1[:, :, full_ids1]
                        # print('Updating: ', update_0, update_1, full_ids0.shape, full_ids1.shape)

        acc_corr = torch.zeros(size=[], device=desc0.device) + 0
        acc_corr_total = torch.zeros(size=[], device=desc0.device) + 1
        acc_incorr = torch.zeros(size=[], device=desc0.device) + 0
        acc_incorr_total = torch.zeros(size=[], device=desc0.device) + 1
        output = {
            'scores': [pred_score],
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'acc_corr': [acc_corr],
            'acc_incorr': [acc_incorr],
            'total_acc_corr': [acc_corr_total],
            'total_acc_incorr': [acc_incorr_total],
        }

        return output

    def produce_matches_test_R50(self, data, p=0.2, mscore_th=0.1, uncertainty_ratio=1., **kwargs):
        # print('In produce_matches_test....')
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        scores0, scores1 = data['scores0'], data['scores1']

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        nK0 = desc0.shape[-1]
        nK1 = desc1.shape[-1]
        nI = self.config['n_layers']
        nB = desc0.shape[0]

        assert nB == 1

        dev = desc0.device
        global_ids0 = torch.arange(0, desc0.shape[-1], device=dev, requires_grad=False)
        global_ids1 = torch.arange(0, desc1.shape[-1], device=dev, requires_grad=False)
        full_ids0 = global_ids0
        full_ids1 = global_ids1
        sample_ids0 = []
        sample_ids1 = []

        M00 = None
        M01 = None
        M10 = None
        M11 = None
        prob00 = None
        prob01 = None
        prob11 = None
        prob10 = None

        all_indices0 = []
        all_mscores0 = []

        for ni in range(nI):
            # for ni in range(5):
            # print('\n....................Process ni: ', ni, desc0[0, 0, 0:5], desc1[0, 0, 0:5], desc0.shape,
            #       desc1.shape)
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0,
                           prob=None if prob00 is None else prob00,
                           M=None if M00 is None else M00)
            prob00 = layer.prob
            delta1 = layer(desc1, desc1,
                           prob=None if prob11 is None else prob11,
                           M=None if M11 is None else M11)
            prob11 = layer.prob
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # print('self desc0: ', ni, desc0[:, 0, 0:5])
            # print('self desc1: ', ni, desc1[:, 0, 0:5])

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1,
                           prob=None if prob10 is None else prob10,
                           M=None if M10 is None or ni == 3 else M10)
            prob10 = layer.prob
            delta1 = layer(desc1, desc0,
                           prob=None if prob01 is None else prob01,
                           M=None if M01 is None or ni == 3 else M01)
            prob01 = layer.prob
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # print('cross desc0: ', ni, desc0[:, 0, 0:5])
            # print('cross desc1: ', ni, desc1[:, 0, 0:5])

            if not self.multi_proj:
                mdesc0 = self.final_proj(desc0)
                mdesc1 = self.final_proj(desc1)
            else:
                mdesc0 = self.final_proj[ni](desc0)
                mdesc1 = self.final_proj[ni](desc1)

            if ni < self.first_it_to_update:
                # if ni < 3:
                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
                all_indices0.append(indices0)
                # all_indices1.append(indices1)
                all_mscores0.append(mscores0)
                # all_mscores1.append(mscores1)

                # keep all samples
                sample_ids0.append(global_ids0)
                sample_ids1.append(global_ids1)

            else:  # pooling based on the results of 1
                bi = 0
                full_indices0 = torch.zeros((nB, nK0), device=dev, requires_grad=False).long() - 1
                full_mscores0 = torch.zeros((nB, nK0), device=dev, requires_grad=False)

                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
                indices0 = indices0[bi]
                mscores0 = mscores0[bi]
                valid0 = (indices0 >= 0)

                full_indices0[bi, sample_ids0[-1][valid0]] = sample_ids1[-1][indices0[valid0]]
                full_mscores0[bi, sample_ids0[-1]] = mscores0

                all_indices0.append(full_indices0)
                all_mscores0.append(full_mscores0)

                perform_updating = self.sharing_layers[2 * ni]
                if not perform_updating:
                    sample_ids0.append(sample_ids0[-1])
                    sample_ids1.append(sample_ids1[-1])
                else:
                    sum_prob00 = torch.sum(prob00, dim=1).sum(dim=1)
                    sum_prob01 = torch.sum(prob01, dim=1).sum(dim=1)
                    sum_prob10 = torch.sum(prob10, dim=1).sum(dim=1)
                    sum_prob11 = torch.sum(prob11, dim=1).sum(dim=1)

                    norm_prob00 = sum_prob00 / torch.sum(sum_prob00, dim=1, keepdim=True)
                    norm_prob01 = sum_prob01 / torch.sum(sum_prob01, dim=1, keepdim=True)
                    norm_prob10 = sum_prob10 / torch.sum(sum_prob10, dim=1, keepdim=True)
                    norm_prob11 = sum_prob11 / torch.sum(sum_prob11, dim=1, keepdim=True)

                    # print('sum_p00: ', sum_prob00[0, :5], sum_prob00.shape)
                    # print('sum_p01: ', sum_prob01[0, :5], sum_prob01.shape)
                    # print('sum_p11: ', sum_prob11[0, :5], sum_prob11.shape)
                    # print('sum_p10: ', sum_prob10[0, :5], sum_prob10.shape)
                    with torch.no_grad():
                        if self.n_min_tokens > 0 and sample_ids0[-1].shape[-1] <= self.n_min_tokens:
                            update_0 = False
                        else:
                            update_0 = True

                        if self.n_min_tokens > 0 and sample_ids1[-1].shape[-1] <= self.n_min_tokens:
                            update_1 = False
                        else:
                            update_1 = True

                        if update_0:
                            md_prob00 = torch.median(norm_prob00[bi])
                            md_prob01 = torch.median(norm_prob01[bi])
                            aug_ids00 = torch.where(norm_prob00[bi] >= md_prob00)[0]
                            aug_ids01 = torch.where(norm_prob01[bi] >= md_prob01)[0]
                            full_ids0 = torch.unique(torch.hstack([aug_ids00, aug_ids01]))

                            sample_ids0.append(sample_ids0[-1][full_ids0])
                        else:
                            sample_ids0.append(sample_ids0[-1])

                        if update_1:
                            # _, pids1 = torch.where(
                            #     torch.sum(pred_score[:, :-1, :-1], dim=1) >= mscore_th * uncertainty_ratio)
                            # if pids1.shape[-1] > 0:
                            md_prob10 = torch.median(norm_prob10[bi])
                            md_prob11 = torch.median(norm_prob11[bi])
                            aug_ids10 = torch.where(norm_prob10[bi] >= md_prob10)[0]
                            aug_ids11 = torch.where(norm_prob11[bi] >= md_prob11)[0]
                            full_ids1 = torch.unique(torch.hstack([aug_ids10, aug_ids11]))

                            # print('full_ids1: ', desc1[bi, :].shape, full_ids1.shape)
                            sample_ids1.append(sample_ids1[-1][full_ids1])
                        else:
                            sample_ids1.append(sample_ids1[-1])

                        # update probability
                        if update_0 and update_1:
                            desc0 = desc0[:, :, full_ids0]
                            desc1 = desc1[:, :, full_ids1]

                        elif update_0 and (not update_1):
                            desc0 = desc0[:, :, full_ids0]
                        elif (not update_0) and update_1:
                            desc1 = desc1[:, :, full_ids1]
                        # print('Updating: ', update_0, update_1, full_ids0.shape, full_ids1.shape)

        acc_corr = torch.zeros(size=[], device=desc0.device) + 0
        acc_corr_total = torch.zeros(size=[], device=desc0.device) + 1
        acc_incorr = torch.zeros(size=[], device=desc0.device) + 0
        acc_incorr_total = torch.zeros(size=[], device=desc0.device) + 1
        output = {
            'scores': [pred_score],
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'acc_corr': [acc_corr],
            'acc_incorr': [acc_incorr],
            'total_acc_corr': [acc_corr_total],
            'total_acc_incorr': [acc_incorr_total],
        }

        return output

    def produce_matches_vis(self, data, p=0.2, mscore_th=0.1, uncertainty_ratio=1., test_it=-1, **kwargs):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        scores0, scores1 = data['scores0'], data['scores1']

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        nK0 = desc0.shape[-1]
        nK1 = desc1.shape[-1]
        nI = self.config['n_layers']
        nB = desc0.shape[0]

        assert nB == 1

        dev = desc0.device
        global_ids0 = torch.arange(0, desc0.shape[-1], device=dev, requires_grad=False)
        global_ids1 = torch.arange(0, desc1.shape[-1], device=dev, requires_grad=False)
        full_ids0 = global_ids0
        full_ids1 = global_ids1
        sample_ids0 = []
        sample_ids1 = []

        M00 = None
        M01 = None
        M10 = None
        M11 = None
        prob00 = None
        prob01 = None
        prob11 = None
        prob10 = None

        all_indices0 = []
        all_mscores0 = []

        all_prob00 = []
        all_prob01 = []
        all_prob11 = []
        all_prob10 = []

        if test_it >= 1 and test_it <= nI:
            nI = test_it
        for ni in range(nI):
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0,
                           prob=None if prob00 is None else prob00,
                           M=None if M00 is None else M00)
            prob00 = layer.prob

            delta1 = layer(desc1, desc1,
                           prob=None if prob11 is None else prob11,
                           M=None if M11 is None else M11)
            prob11 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1,
                           prob=None if prob10 is None else prob10,
                           M=None if M10 is None or ni == 3 else M10)
            prob10 = layer.prob

            delta1 = layer(desc1, desc0,
                           prob=None if prob01 is None else prob01,
                           M=None if M01 is None or ni == 3 else M01)
            prob01 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            all_prob00.append(prob00)
            all_prob11.append(prob11)
            all_prob10.append(prob10)
            all_prob01.append(prob01)

            if not self.multi_proj:
                mdesc0 = self.final_proj(desc0)
                mdesc1 = self.final_proj(desc1)
            else:
                mdesc0 = self.final_proj[ni](desc0)
                mdesc1 = self.final_proj[ni](desc1)

            if ni < self.first_it_to_update:
                # if ni < 3:
                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
                all_indices0.append(indices0)
                # all_indices1.append(indices1)
                all_mscores0.append(mscores0)
                # all_mscores1.append(mscores1)

                # keep all samples
                sample_ids0.append(global_ids0)
                sample_ids1.append(global_ids1)

            else:  # pooling based on the results of 1
                bi = 0
                full_indices0 = torch.zeros((nB, nK0), device=dev, requires_grad=False).long() - 1
                full_mscores0 = torch.zeros((nB, nK0), device=dev, requires_grad=False)

                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
                indices0 = indices0[bi]
                mscores0 = mscores0[bi]
                valid0 = (indices0 >= 0)

                full_indices0[bi, sample_ids0[-1][valid0]] = sample_ids1[-1][indices0[valid0]]
                full_mscores0[bi, sample_ids0[-1]] = mscores0

                all_indices0.append(full_indices0)
                all_mscores0.append(full_mscores0)

                perform_updating = self.sharing_layers[2 * ni]
                if not perform_updating:
                    sample_ids0.append(sample_ids0[-1])
                    sample_ids1.append(sample_ids1[-1])
                else:
                    sum_prob00 = torch.sum(prob00, dim=1).sum(dim=1)
                    sum_prob01 = torch.sum(prob01, dim=1).sum(dim=1)
                    sum_prob10 = torch.sum(prob10, dim=1).sum(dim=1)
                    sum_prob11 = torch.sum(prob11, dim=1).sum(dim=1)

                    norm_prob00 = sum_prob00 / torch.sum(sum_prob00, dim=1, keepdim=True)
                    norm_prob01 = sum_prob01 / torch.sum(sum_prob01, dim=1, keepdim=True)
                    norm_prob10 = sum_prob10 / torch.sum(sum_prob10, dim=1, keepdim=True)
                    norm_prob11 = sum_prob11 / torch.sum(sum_prob11, dim=1, keepdim=True)

                    with torch.no_grad():
                        if self.n_min_tokens > 0 and sample_ids0[-1].shape[-1] <= self.n_min_tokens:
                            update_0 = False
                        else:
                            update_0 = True

                        if self.n_min_tokens > 0 and sample_ids1[-1].shape[-1] <= self.n_min_tokens:
                            update_1 = False
                        else:
                            update_1 = True

                        if update_0:
                            _, pids0 = torch.where(
                                torch.sum(pred_score[:, :-1, :-1], dim=-1) >= mscore_th * uncertainty_ratio)
                            # print('pids0: ', ni, pids0)

                            if pids0.shape[-1] > 0:
                                md_prob00 = torch.median(norm_prob00[bi][pids0])
                                md_prob01 = torch.median(norm_prob01[bi][pids0])
                                aug_ids00 = torch.where(norm_prob00[bi] >= md_prob00)[0]
                                aug_ids01 = torch.where(norm_prob01[bi] >= md_prob01)[0]
                                full_ids0 = torch.unique(torch.hstack([pids0, aug_ids00, aug_ids01]))

                                sample_ids0.append(sample_ids0[-1][full_ids0])
                        else:
                            sample_ids0.append(sample_ids0[-1])

                        if update_1:
                            _, pids1 = torch.where(
                                torch.sum(pred_score[:, :-1, :-1], dim=1) >= mscore_th * uncertainty_ratio)
                            if pids1.shape[-1] > 0:
                                md_prob10 = torch.median(norm_prob10[bi][pids1])
                                md_prob11 = torch.median(norm_prob11[bi][pids1])
                                aug_ids10 = torch.where(norm_prob10[bi] >= md_prob10)[0]
                                aug_ids11 = torch.where(norm_prob11[bi] >= md_prob11)[0]
                                full_ids1 = torch.unique(torch.hstack([pids1, aug_ids10, aug_ids11]))

                                # print('full_ids1: ', desc1[bi, :].shape, full_ids1.shape)
                                sample_ids1.append(sample_ids1[-1][full_ids1])
                        else:
                            sample_ids1.append(sample_ids1[-1])

                        # update probability
                        if update_0 and update_1:
                            desc0 = desc0[:, :, full_ids0]
                            desc1 = desc1[:, :, full_ids1]

                        elif update_0 and (not update_1):
                            desc0 = desc0[:, :, full_ids0]
                        elif (not update_0) and update_1:
                            desc1 = desc1[:, :, full_ids1]

                        # print('Updating: ', update_0, update_1, full_ids0.shape, full_ids1.shape)
        output = {
            'scores': [pred_score],
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'sample_ids0': sample_ids0,
            'sample_ids1': sample_ids1,
            'prob00': all_prob00,
            'prob11': all_prob11,
            'prob01': all_prob01,
            'prob10': all_prob10,
        }

        return output

    def produce_matches_vis_r50(self, data, p=0.2, mscore_th=0.1, uncertainty_ratio=1., test_it=-1, **kwargs):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        scores0, scores1 = data['scores0'], data['scores1']

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        nK0 = desc0.shape[-1]
        nK1 = desc1.shape[-1]
        nI = self.config['n_layers']
        nB = desc0.shape[0]

        assert nB == 1

        dev = desc0.device
        global_ids0 = torch.arange(0, desc0.shape[-1], device=dev, requires_grad=False)
        global_ids1 = torch.arange(0, desc1.shape[-1], device=dev, requires_grad=False)
        full_ids0 = global_ids0
        full_ids1 = global_ids1
        sample_ids0 = []
        sample_ids1 = []

        M00 = None
        M01 = None
        M10 = None
        M11 = None
        prob00 = None
        prob01 = None
        prob11 = None
        prob10 = None

        all_indices0 = []
        all_mscores0 = []

        all_prob00 = []
        all_prob01 = []
        all_prob11 = []
        all_prob10 = []

        if test_it >= 1 and test_it <= nI:
            nI = test_it
        for ni in range(nI):
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0,
                           prob=None if prob00 is None else prob00,
                           M=None if M00 is None else M00)
            prob00 = layer.prob

            delta1 = layer(desc1, desc1,
                           prob=None if prob11 is None else prob11,
                           M=None if M11 is None else M11)
            prob11 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1,
                           prob=None if prob10 is None else prob10,
                           M=None if M10 is None or ni == 3 else M10)
            prob10 = layer.prob

            delta1 = layer(desc1, desc0,
                           prob=None if prob01 is None else prob01,
                           M=None if M01 is None or ni == 3 else M01)
            prob01 = layer.prob

            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            all_prob00.append(prob00)
            all_prob11.append(prob11)
            all_prob10.append(prob10)
            all_prob01.append(prob01)

            if not self.multi_proj:
                mdesc0 = self.final_proj(desc0)
                mdesc1 = self.final_proj(desc1)
            else:
                mdesc0 = self.final_proj[ni](desc0)
                mdesc1 = self.final_proj[ni](desc1)

            if ni < self.first_it_to_update:
                # if ni < 3:
                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
                all_indices0.append(indices0)
                # all_indices1.append(indices1)
                all_mscores0.append(mscores0)
                # all_mscores1.append(mscores1)

                # keep all samples
                sample_ids0.append(global_ids0)
                sample_ids1.append(global_ids1)

            else:  # pooling based on the results of 1
                bi = 0
                full_indices0 = torch.zeros((nB, nK0), device=dev, requires_grad=False).long() - 1
                full_mscores0 = torch.zeros((nB, nK0), device=dev, requires_grad=False)

                dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                dist = dist / self.config['descriptor_dim'] ** .5
                # pred_score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
                pred_score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

                indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=pred_score, p=p)
                indices0 = indices0[bi]
                mscores0 = mscores0[bi]
                valid0 = (indices0 >= 0)

                full_indices0[bi, sample_ids0[-1][valid0]] = sample_ids1[-1][indices0[valid0]]
                full_mscores0[bi, sample_ids0[-1]] = mscores0

                all_indices0.append(full_indices0)
                all_mscores0.append(full_mscores0)

                perform_updating = self.sharing_layers[2 * ni]
                if not perform_updating:
                    sample_ids0.append(sample_ids0[-1])
                    sample_ids1.append(sample_ids1[-1])
                else:
                    sum_prob00 = torch.sum(prob00, dim=1).sum(dim=1)
                    sum_prob01 = torch.sum(prob01, dim=1).sum(dim=1)
                    sum_prob10 = torch.sum(prob10, dim=1).sum(dim=1)
                    sum_prob11 = torch.sum(prob11, dim=1).sum(dim=1)

                    norm_prob00 = sum_prob00 / torch.sum(sum_prob00, dim=1, keepdim=True)
                    norm_prob01 = sum_prob01 / torch.sum(sum_prob01, dim=1, keepdim=True)
                    norm_prob10 = sum_prob10 / torch.sum(sum_prob10, dim=1, keepdim=True)
                    norm_prob11 = sum_prob11 / torch.sum(sum_prob11, dim=1, keepdim=True)

                    with torch.no_grad():
                        if self.n_min_tokens > 0 and sample_ids0[-1].shape[-1] <= self.n_min_tokens:
                            update_0 = False
                        else:
                            update_0 = True

                        if self.n_min_tokens > 0 and sample_ids1[-1].shape[-1] <= self.n_min_tokens:
                            update_1 = False
                        else:
                            update_1 = True

                        if update_0:
                            k0 = norm_prob00.shape[1] // 2
                            _, aug_ids00 = torch.topk(norm_prob00[bi], largest=True, k=k0)
                            _, aug_ids01 = torch.topk(norm_prob01[bi], largest=True, k=k0)
                            print(k0, aug_ids00)
                            full_ids0 = torch.unique(torch.hstack([aug_ids00, aug_ids01]))
                            sample_ids0.append(sample_ids0[-1][full_ids0])
                        else:
                            sample_ids0.append(sample_ids0[-1])

                        if update_1:
                            k1 = norm_prob11.shape[1] // 2
                            _, aug_ids11 = torch.topk(norm_prob11[bi], largest=True, k=k1)
                            _, aug_ids10 = torch.topk(norm_prob10[bi], largest=True, k=k1)
                            full_ids1 = torch.unique(torch.hstack([aug_ids11, aug_ids10]))
                            sample_ids1.append(sample_ids1[-1][full_ids1])
                        else:
                            sample_ids1.append(sample_ids1[-1])

                        # update probability
                        if update_0 and update_1:
                            desc0 = desc0[:, :, full_ids0]
                            desc1 = desc1[:, :, full_ids1]

                        elif update_0 and (not update_1):
                            desc0 = desc0[:, :, full_ids0]
                        elif (not update_0) and update_1:
                            desc1 = desc1[:, :, full_ids1]

                        # print('Updating: ', update_0, update_1, full_ids0.shape, full_ids1.shape)
        output = {
            'scores': [pred_score],
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'sample_ids0': sample_ids0,
            'sample_ids1': sample_ids1,
            'prob00': all_prob00,
            'prob11': all_prob11,
            'prob01': all_prob01,
            'prob10': all_prob10,
        }

        return output

    def test_time(self, data, p=0.2, mscore_th=0.1, uncertainty_ratio=1., **kwargs):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        scores0, scores1 = data['scores0'], data['scores1']
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        nK0 = desc0.shape[-1]
        nK1 = desc1.shape[-1]
        nI = self.config['n_layers']
        nB = desc0.shape[0]

        assert nB == 1

        dev = desc0.device
        global_ids0 = torch.arange(0, desc0.shape[-1], device=dev, requires_grad=False)
        global_ids1 = torch.arange(0, desc1.shape[-1], device=dev, requires_grad=False)
        full_ids0 = global_ids0
        full_ids1 = global_ids1
        sample_ids0 = []
        sample_ids1 = []

        M00 = None
        M01 = None
        M10 = None
        M11 = None
        prob00 = None
        prob01 = None
        prob11 = None
        prob10 = None

        for ni in range(nI):
            layer = self.gnn.layers[ni * 2]
            delta0 = layer(desc0, desc0,
                           prob=None if prob00 is None else prob00,
                           M=None if M00 is None else M00)
            prob00 = layer.prob
            delta1 = layer(desc1, desc1,
                           prob=None if prob11 is None else prob11,
                           M=None if M11 is None else M11)
            prob11 = layer.prob
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            # cross attention then
            layer = self.gnn.layers[ni * 2 + 1]
            delta0 = layer(desc0, desc1,
                           prob=None if prob10 is None else prob10,
                           M=None if M10 is None or ni == 3 else M10)
            prob10 = layer.prob
            delta1 = layer(desc1, desc0,
                           prob=None if prob01 is None else prob01,
                           M=None if M01 is None or ni == 3 else M01)
            prob01 = layer.prob
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1

            if ni < self.first_it_to_update:
                # keep all samples
                sample_ids0.append(global_ids0)
                sample_ids1.append(global_ids1)

            else:  # pooling based on the results of 1
                perform_updating = (self.sharing_layers[2 * ni] and ni in [3])
                if not perform_updating:
                    sample_ids0.append(sample_ids0[-1])
                    sample_ids1.append(sample_ids1[-1])
                else:
                    if not self.multi_proj:
                        mdesc0 = self.final_proj(desc0)
                        mdesc1 = self.final_proj(desc1)
                    else:
                        mdesc0 = self.final_proj[ni](desc0)
                        mdesc1 = self.final_proj[ni](desc1)

                    bi = 0
                    dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
                    dist = dist / self.config['descriptor_dim'] ** .5
                    pred_score = self.compute_score(dist=dist, dustbin=self.bin_score,
                                                    iteration=self.sinkhorn_iterations)

                    sum_prob00 = torch.sum(prob00, dim=1).sum(dim=1)
                    sum_prob01 = torch.sum(prob01, dim=1).sum(dim=1)
                    sum_prob10 = torch.sum(prob10, dim=1).sum(dim=1)
                    sum_prob11 = torch.sum(prob11, dim=1).sum(dim=1)

                    norm_prob00 = sum_prob00 / torch.sum(sum_prob00, dim=1, keepdim=True)
                    norm_prob01 = sum_prob01 / torch.sum(sum_prob01, dim=1, keepdim=True)
                    norm_prob10 = sum_prob10 / torch.sum(sum_prob10, dim=1, keepdim=True)
                    norm_prob11 = sum_prob11 / torch.sum(sum_prob11, dim=1, keepdim=True)

                    with torch.no_grad():
                        if self.n_min_tokens > 0 and sample_ids0[-1].shape[-1] <= self.n_min_tokens:
                            update_0 = False
                        else:
                            update_0 = True

                        if self.n_min_tokens > 0 and sample_ids1[-1].shape[-1] <= self.n_min_tokens:
                            update_1 = False
                        else:
                            update_1 = True

                        if update_0:
                            _, pids0 = torch.where(
                                torch.sum(pred_score[:, :-1, :-1], dim=-1) >= mscore_th * uncertainty_ratio)
                            if pids0.shape[-1] > 100:
                                md_prob00 = torch.median(norm_prob00[bi][pids0])
                                md_prob01 = torch.median(norm_prob01[bi][pids0])
                                aug_ids00 = torch.where(norm_prob00[bi] >= md_prob00)[0]
                                aug_ids01 = torch.where(norm_prob01[bi] >= md_prob01)[0]
                                full_ids0 = torch.unique(torch.hstack([pids0, aug_ids00, aug_ids01]))
                                sample_ids0.append(sample_ids0[-1][full_ids0])
                                # print('full_ids0: ', full_ids0.shape, pids0.shape, aug_ids00.shape, aug_ids01.shape)
                            # mask_pids0 = (torch.sum(pred_score[0, :-1, :-1], dim=-1) >= mscore_th * uncertainty_ratio)
                            # if torch.sum(mask_pids0) > 0:
                            #     md_prob00 = torch.median(norm_prob00[0][mask_pids0])
                            #     md_prob01 = torch.median(norm_prob01[0][mask_pids0])
                            #     mask_aug_ids00 = (norm_prob00[0] >= md_prob00)
                            #     mask_aug_ids01 = (norm_prob01[0] >= md_prob01)
                            #     mask_full_ids0 = mask_pids0 + mask_aug_ids00 + mask_aug_ids01
                            #     full_ids0 = torch.where(mask_full_ids0 == True)[0]
                            #     sample_ids0.append(sample_ids0[-1][full_ids0])

                        else:
                            sample_ids0.append(sample_ids0[-1])

                        if update_1:
                            _, pids1 = torch.where(
                                torch.sum(pred_score[:, :-1, :-1], dim=1) >= mscore_th * uncertainty_ratio)
                            if pids1.shape[-1] > 100:
                                md_prob10 = torch.median(norm_prob10[bi][pids1])
                                md_prob11 = torch.median(norm_prob11[bi][pids1])
                                aug_ids10 = torch.where(norm_prob10[bi] >= md_prob10)[0]
                                aug_ids11 = torch.where(norm_prob11[bi] >= md_prob11)[0]
                                full_ids1 = torch.unique(torch.hstack([pids1, aug_ids10, aug_ids11]))

                                sample_ids1.append(sample_ids1[-1][full_ids1])

                                # print('full_ids1: ', full_ids1.shape)
                            # mask_pids1 = (torch.sum(pred_score[0, :-1, :-1], dim=0) >= mscore_th * uncertainty_ratio)
                            # if torch.sum(mask_pids1) > 0:
                            #     md_prob10 = torch.median(norm_prob10[0][mask_pids1])
                            #     md_prob11 = torch.median(norm_prob11[0][mask_pids1])
                            #     mask_aug_ids10 = (norm_prob10[0] >= md_prob10)
                            #     mask_aug_ids11 = (norm_prob11[bi] >= md_prob11)
                            #     mask_full_ids1 = mask_pids1 + mask_aug_ids10 + mask_aug_ids11
                            #     full_ids1 = torch.where(mask_full_ids1 == True)[0]
                            #     sample_ids1.append(sample_ids1[-1][full_ids1])
                        else:
                            sample_ids1.append(sample_ids1[-1])

                        # update descriptor
                        if update_0 and update_1:
                            desc0 = desc0[:, :, full_ids0].contiguous()
                            desc1 = desc1[:, :, full_ids1].contiguous()

                        elif update_0 and (not update_1):
                            desc0 = desc0[:, :, full_ids0].contiguous()
                        elif (not update_0) and update_1:
                            desc1 = desc1[:, :, full_ids1].contiguous()

                        # print('ni:, update_0 / 1', ni, update_0, update_1, full_ids0.shape, full_ids1.shape)
        output = {
            'pred_score': pred_score,
        }

        return output

    def forward_one_layer_old(self, desc0, desc1, M0, M1, layer_i):
        bs = desc0.shape[0]  # should be 1
        N0 = desc0.shape[2]
        N1 = desc1.shape[2]
        dev = desc0.device

        layer = self.gnn.layers[layer_i]
        name = self.gnn.names[layer_i]
        pool_size = self.pool_sizes[layer_i]

        if layer_i == 0:
            self.self_indexes0 = torch.arange(0, N0, device=dev)[None].repeat(bs, 1)
            self.cross_indexes0 = torch.arange(0, N0, device=dev)[None].repeat(bs, 1)
            self.self_indexes1 = torch.arange(0, N1, device=dev)[None].repeat(bs, 1)
            self.cross_indexes1 = torch.arange(0, N1, device=dev)[None].repeat(bs, 1)

            self.last_self_packs0 = None
            self.last_self_packs1 = None
            self.last_cross_packs0 = None
            self.last_cross_packs1 = None

        if name == 'cross':
            if pool_size > 0:
                ds_desc0, cross_indexes01 = self.gnn.ada_pooling_index(X=desc0, global_indexes=self.cross_indexes0,
                                                                       prob=self.cross_prob0,
                                                                       pool_size=pool_size)
                ds_desc1, cross_indexes10 = self.gnn.ada_pooling_index(X=desc1, global_indexes=self.cross_indexes1,
                                                                       prob=self.cross_prob1,
                                                                       pool_size=pool_size)

                self.cross_indexes0 = cross_indexes01
                self.cross_indexes1 = cross_indexes10
            else:
                ds_desc0 = desc0
                ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc1, self.cross_prob1, M=None)
            # self.cross_attns10.append(layer.prob)
            self.cross_prob1 = layer.prob
            delta1 = layer(desc1, ds_desc0, self.cross_prob0, M=None)
            # self.cross_attns01.append(layer.prob)
            self.cross_prob0 = layer.prob

        elif name == 'self':
            if pool_size > 0:
                ds_desc0, self_indexes00 = self.gnn.ada_pooling_index(X=desc0, global_indexes=self.self_indexes0,
                                                                      prob=self.self_prob0,
                                                                      pool_size=pool_size)
                ds_desc1, self_indexes11 = self.gnn.ada_pooling_index(X=desc1, global_indexes=self.self_indexes1,
                                                                      prob=self.self_prob1,
                                                                      pool_size=pool_size)

                self.self_indexes0 = self_indexes00
                self.self_indexes1 = self_indexes11

            else:
                ds_desc0 = desc0
                ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc0, self.self_prob0, M=None)
            # self.self_attns00.append(layer.prob)
            self.self_prob0 = layer.prob
            delta1 = layer(desc1, ds_desc1, self.self_prob1, M=None)
            # self.self_attns11.append(layer.prob)
            self.self_prob1 = layer.prob

        return desc0 + delta0, desc1 + delta1

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        layer = self.gnn.layers[layer_i]
        name = self.gnn.names[layer_i]

        if name == 'cross':
            ds_desc0 = desc0
            ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc1, self.cross_prob1, M=None)
            self.cross_prob1 = layer.prob
            delta1 = layer(desc1, ds_desc0, self.cross_prob0, M=None)
            self.cross_prob0 = layer.prob

        elif name == 'self':
            ds_desc0 = desc0
            ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc0, self.self_prob0, M=None)
            self.self_prob0 = layer.prob
            delta1 = layer(desc1, ds_desc1, self.self_prob1, M=None)
            self.self_prob1 = layer.prob

        return desc0 + delta0, desc1 + delta1

    def pool(self, pred_score, prob00, prob01, prob11, prob10, mscore_th=0.1, uncertainty_ratio=1.0,
             n_min_tokens=256):
        n0 = pred_score.shape[1]
        n1 = pred_score.shape[2]

        sum_prob00 = torch.sum(prob00, dim=1).sum(dim=1)
        sum_prob01 = torch.sum(prob01, dim=1).sum(dim=1)
        sum_prob10 = torch.sum(prob10, dim=1).sum(dim=1)
        sum_prob11 = torch.sum(prob11, dim=1).sum(dim=1)

        norm_prob00 = sum_prob00 / torch.sum(sum_prob00, dim=1, keepdim=True)
        norm_prob01 = sum_prob01 / torch.sum(sum_prob01, dim=1, keepdim=True)
        norm_prob10 = sum_prob10 / torch.sum(sum_prob10, dim=1, keepdim=True)
        norm_prob11 = sum_prob11 / torch.sum(sum_prob11, dim=1, keepdim=True)

        if n_min_tokens > 0 and n0 <= n_min_tokens:
            update_0 = False
        else:
            update_0 = True
        if n_min_tokens > 0 and n1 <= n_min_tokens:
            update_1 = False
        else:
            update_1 = True

        bi = 0
        if update_0:
            _, pids0 = torch.where(
                torch.sum(pred_score[:, :-1, :-1], dim=-1) >= mscore_th * uncertainty_ratio)
            if pids0.shape[-1] > 0:
                md_prob00 = torch.median(norm_prob00[bi][pids0])
                md_prob01 = torch.median(norm_prob01[bi][pids0])
                aug_ids00 = torch.where(norm_prob00[bi] >= md_prob00)[0]
                aug_ids01 = torch.where(norm_prob01[bi] >= md_prob01)[0]
                full_ids0 = torch.unique(torch.hstack([pids0, aug_ids00, aug_ids01]))
            else:
                full_ids0 = None
        else:
            full_ids0 = None

        if update_1:
            _, pids1 = torch.where(
                torch.sum(pred_score[:, :-1, :-1], dim=1) >= mscore_th * uncertainty_ratio)
            if pids1.shape[-1] > 0:
                md_prob10 = torch.median(norm_prob10[bi][pids1])
                md_prob11 = torch.median(norm_prob11[bi][pids1])
                aug_ids10 = torch.where(norm_prob10[bi] >= md_prob10)[0]
                aug_ids11 = torch.where(norm_prob11[bi] >= md_prob11)[0]
                full_ids1 = torch.unique(torch.hstack([pids1, aug_ids10, aug_ids11]))
            else:
                full_ids1 = None
        else:
            full_ids1 = None

        return full_ids0, full_ids1

    def pool_pose_attention(self, x0_3d, x1_3d, pred_E, inlier_ratio, pred_score, prob00, prob01, prob11,
                            prob10,
                            mscore_th=0.1,
                            uncertainty_ratio=1.0,
                            epi_th=1e-5,
                            with_attn=True,
                            n_candidate=3,
                            n_min_tokens=256):
        n0 = pred_score.shape[1]
        n1 = pred_score.shape[2]

        if n_min_tokens > 0 and n0 <= n_min_tokens:
            update_0 = False
        else:
            update_0 = True
        if n_min_tokens > 0 and n1 <= n_min_tokens:
            update_1 = False
        else:
            update_1 = True

        if (not update_0) and (not update_1):
            return None, None

        print('pred_score: ', pred_score.shape)
        with torch.no_grad():
            _, can_ids0 = torch.topk(pred_score[:, :-1, :-1], dim=2, k=n_candidate, largest=True)
            _, can_ids1 = torch.topk(pred_score[:, :-1, :-1], dim=1, k=n_candidate, largest=True)

            dist = _sampson_dist_general(F=pred_E, X=x0_3d, Y=x1_3d)
            dist0 = torch.gather(dist, index=can_ids0, dim=2)
            min_dist0, _ = torch.min(dist0, dim=2)
            dist1 = torch.gather(dist.transpose(1, 2), index=can_ids1.transpose(1, 2), dim=2)
            min_dist1, _ = torch.min(dist1, dim=2)
            geo_pids0 = torch.where(min_dist0 <= epi_th)[1]
            geo_pids1 = torch.where(min_dist1 <= epi_th)[1]

        print('geo_ids0: ', geo_pids0.shape)
        print('geo_ids1: ', geo_pids1.shape)

        if with_attn:
            sum_prob00 = torch.sum(prob00, dim=1).sum(dim=1)
            sum_prob01 = torch.sum(prob01, dim=1).sum(dim=1)
            sum_prob10 = torch.sum(prob10, dim=1).sum(dim=1)
            sum_prob11 = torch.sum(prob11, dim=1).sum(dim=1)
            norm_prob00 = sum_prob00 / torch.sum(sum_prob00, dim=1, keepdim=True)
            norm_prob01 = sum_prob01 / torch.sum(sum_prob01, dim=1, keepdim=True)
            norm_prob10 = sum_prob10 / torch.sum(sum_prob10, dim=1, keepdim=True)
            norm_prob11 = sum_prob11 / torch.sum(sum_prob11, dim=1, keepdim=True)

        bi = 0
        if update_0:
            _, pids0 = torch.where(
                torch.sum(pred_score[:, :-1, :-1], dim=-1) >= mscore_th * inlier_ratio)
            if pids0.shape[-1] > 0:
                if with_attn:
                    md_prob00 = torch.median(norm_prob00[bi][pids0])
                    md_prob01 = torch.median(norm_prob01[bi][pids0])
                    attn_ids00 = torch.where(norm_prob00[bi] >= md_prob00)[0]
                    attn_ids01 = torch.where(norm_prob01[bi] >= md_prob01)[0]
                    full_ids0 = torch.unique(torch.hstack([geo_pids0, pids0, attn_ids00, attn_ids01]))
                else:
                    full_ids0 = torch.unique(torch.hstack([geo_pids0, pids0]))
            else:
                full_ids0 = geo_pids0
        else:
            full_ids0 = None

        if update_1:
            _, pids1 = torch.where(
                torch.sum(pred_score[:, :-1, :-1], dim=1) >= mscore_th * uncertainty_ratio)
            if pids1.shape[-1] > 0:
                if with_attn:
                    md_prob10 = torch.median(norm_prob10[bi][pids1])
                    md_prob11 = torch.median(norm_prob11[bi][pids1])
                    attn_ids10 = torch.where(norm_prob10[bi] >= md_prob10)[0]
                    attn_ids11 = torch.where(norm_prob11[bi] >= md_prob11)[0]
                    full_ids1 = torch.unique(torch.hstack([pids1, geo_pids1, attn_ids10, attn_ids11]))
                else:
                    full_ids1 = torch.unique(torch.hstack([geo_pids1, pids1]))
            else:
                full_ids1 = geo_pids1
        else:
            full_ids1 = None

        if full_ids0 is not None and full_ids1 is not None:
            print('full_ids0/1: ', full_ids0.shape, full_ids1.shape)
        return full_ids0, full_ids1

    def run(self, data):
        # used for evaluation of sgmnet
        desc0 = data['desc1']
        desc1 = data['desc2']
        kpts0 = data['x1'][:, :, :2]
        kpts1 = data['x2'][:, :, :2]
        scores0 = data['x1'][:, :, -1]
        scores1 = data['x2'][:, :, -1]

        out = self.produce_matches_test(
            data={
                'descriptors0': desc0,
                'descriptors1': desc1,
                'norm_keypoints0': kpts0,
                'norm_keypoints1': kpts1,
                'scores0': scores0,
                'scores1': scores1,
            },
            p=self.config['match_threshold'])
        # out = self.produce_matches(data=data)
        indices0 = out['indices0'][-1][0]
        # indices1 = out['indices0'][-1][0]
        index0 = torch.where(indices0 >= 0)[0]
        index1 = indices0[index0]

        return {
            'index0': index0,
            'index1': index1,
        }
