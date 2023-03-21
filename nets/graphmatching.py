from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from nets.loss import GraphLoss
from nets.poses import PoseEstimator

from nets.basic_layers import sink_algorithm


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
                # layers.append(nn.InstanceNorm1d(channels, eps=1e-3))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]  # [B, 2, N] + [B, 1, N]
        return self.encoder(torch.cat(inputs, dim=1))


class MultiClusterMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, M=None):
        '''
        :param query: [B, D, N]
        :param key: [B, D, M]
        :param value: [B, D, M]
        :param M: [B, N, M]
        :return:
        '''

        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]  # [B, D, NH, N]
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5

        if M is not None:
            # print('M: ', scores.shape, M.shape, torch.sum(M, dim=2))
            scores = scores * M[:, None, :].expand_as(scores)
        prob = F.softmax(scores, dim=-1)
        x = torch.einsum('bhnm,bdhm->bdhn', prob, value)
        self.prob = torch.mean(prob, dim=1)

        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class ClusterMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, query_labels=None, value_labels=None):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]

        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
        # no matched labels, do exhaustive attention
        if query_labels is None or value_labels is None:
            prob = torch.nn.functional.softmax(scores, dim=-1)
            value = torch.einsum('bhnm,bdhm->bdhn', prob, value)

            self.prob.append(prob)
            return self.merge(value.contiguous().view(batch_dim, self.dim * self.num_heads, -1))

            # return value, prob

        # with matched labels, do label-wise attention
        new_value = torch.zeros_like(value)
        # new_prob = torch.zeros_like(scores)
        for bid in range(batch_dim):
            batch_query_labels = query_labels[bid]
            batch_value_labels = value_labels[bid]
            batch_scores = scores[bid]
            for l in torch.unique(batch_value_labels):
                value_mask = (batch_value_labels == l)
                query_mask = (batch_query_labels == l)
                # print('mask: ', torch.sum(value_mask), torch.sum(query_mask))

                batch_l_value = value[bid][:, :, value_mask]
                if torch.sum(query_mask) == 0:
                    batch_l_prob = F.softmax(batch_scores, dim=-1)
                else:
                    batch_l_prob = F.softmax(batch_scores[:, query_mask, :][:, :, value_mask], dim=-1)

                # print('batch_l_value: ', batch_l_value.shape, batch_l_prob.shape)

                new_batch_l_value = torch.einsum('bhnm,bdhm->bdhn', batch_l_prob[None], batch_l_value[None])

                new_value[bid, :, :, query_mask] = new_batch_l_value

                self.prob.append(scores)
        return self.merge(new_value.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class MultiClusterAttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiClusterMultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward_label(self, x, source, labels_x=None, labels_s=None, confidences_x=None, confidences_s=None):
        message = self.attn(x, source, source, labels_x, labels_s, confidences_x, confidences_s)
        return self.mlp(torch.cat([x, message], dim=1))

    def forward(self, x, source, M=None):
        message = self.attn(x, source, source, M=M)
        return self.mlp(torch.cat([x, message], dim=1))


class MultiClusterAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, pooling_sizes: list = None, keep_ratios: list = None):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiClusterAttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))
        ])
        self.names = layer_names

        if pooling_sizes is None and keep_ratios is None:
            self.use_ada = False
            self.pooling_values = [1] * len(layer_names)
        else:
            self.pooling_sizes = pooling_sizes
            self.keep_ratios = keep_ratios

            if self.pooling_sizes is not None:
                self.pooling_values = self.pooling_sizes
                self.pool_fun = self.update_connections_by_pooling_size
            elif self.keep_ratios is not None:
                self.pooling_values = self.keep_ratios
                self.pool_fun = self.update_connections_by_ratio

    def forward(self, desc0, desc1):
        all_desc0s = []
        all_desc1s = []
        bs = desc0.shape[0]
        for i, (layer, name, pool_value) in enumerate(zip(self.layers, self.names, self.pooling_values)):
            if name == 'cross':
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc1, desc0], dim=0), None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]
            else:
                delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc0, desc1], dim=0), None)
                delta0 = delta[:bs]
                delta1 = delta[bs:]

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)
        return all_desc0s, all_desc1s, []

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        layer = self.layers[layer_i]
        name = self.names[layer_i]
        bs = desc0.shape[0]
        # print(layer_i, name)
        if name == 'cross':
            delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc1, desc0], dim=0),
                          None if M0 is None or M1 is None else torch.cat([M0, M1], dim=0))
            delta0 = delta[:bs]
            delta1 = delta[bs:]
            # delta0 = layer(desc0, desc1, None)
            # delta1 = layer(desc1, desc0, None)
        else:
            delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc0, desc1], dim=0),
                          None if M0 is None or M1 is None else torch.cat([M0, M1], dim=0))
            delta0 = delta[:bs]
            delta1 = delta[bs:]
            # delta0 = layer(desc0, desc0, None)
            # delta1 = layer(desc1, desc1, None)
        return desc0 + delta0, desc1 + delta1

    def update_connections_by_pooling_size(self, last_connections, last_prob, pool_size):
        if last_connections is None:
            return None
        if last_prob is None or pool_size is None:
            return last_connections
        if pool_size == 1:
            return last_connections

        with torch.no_grad():
            # bs = last_connections.shape[0]
            # print('last: ', last_connections.shape)
            current_connections = torch.zeros_like(last_connections)
            k = int(torch.sum(last_connections[0], dim=-1)[0].item() // pool_size)
            # print('k: ', k, last_connections.dtype, last_prob.dtype)
            values, idxs = last_prob.topk(k=k, largest=True, dim=2)
            current_connections = current_connections.scatter(2, idxs, values)
            current_connections[current_connections > 0] = 1
            return current_connections

    def update_connections_by_ratio(self, last_connections, last_prob, ratio):
        if last_connections is None:
            return None
        if last_prob is None or ratio is None:
            return last_connections
        if ratio == 1:
            return last_connections

        with torch.no_grad():
            bs = last_connections.shape[0]
            current_connections = torch.zeros_like(last_connections)
            # idxs = torch.argsort(last_prob, descending=True, dim=2)
            # print('idxs: ', idxs.shape)
            # sorted_prob = torch.zeros_like(last_prob)
            sorted_prob, idxs = torch.sort(last_prob, dim=2, descending=True)
            print('prob: ', sorted_prob.shape, idxs.shape)

            acc_prob = torch.cumsum(sorted_prob, dim=2)
            # acc_prob[(acc_prob < ratio)] += 10
            # print('acc_prob: ', acc_prob.shape)
            # values, idxs2 = torch.min(acc_prob, dim=2)
            current_connections = current_connections.scatter(2, idxs(acc_prob <= ratio),
                                                              sorted_prob(acc_prob <= ratio))
            current_connections[current_connections > 0] = 1
            # for bid in range(bs):
            #     current_connections[bid, idxs[:idxs2[bid] + 1]] = 1


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


torch.autograd.set_detect_anomaly(True)


class GraphMatcher(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,  # [self, cross, self, cross, ...] 9 in total
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
        'with_pose': False,
        'with_label': False,
        'pooling_sizes': None,
        'with_hard_negative': False,
        'neg_margin': 0.2,
        'multi_scale': False,
        'multi_proj': False,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        print('config in GraphMatcher: ', self.config)

        self.neg_ratio = self.config['neg_ratio']
        self.sinkhorn_iterations = self.config['sinkhorn_iterations']
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.gnn = MultiClusterAttentionalGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pooling_sizes=self.config['pooling_sizes'],
        )
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'],
            self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.with_label = self.config['with_label']
        self.with_pose = self.config['with_pose']

        self.match_net = GraphLoss(config=self.config)
        if self.with_pose:
            self.pose_net = PoseEstimator(config=self.config)

    def forward_train(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']
        # print(desc0.shape, desc1.shape, kpts0.shape, kpts1.shape, data['scores0'].shape, data['scores1'].shape)
        # print("norm: ", torch.sum(desc0 ** 2, dim=2)[0, 0:10])

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }
        # Keypoint normalization.
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
        # print('enc: ', enc0.shape, enc1.shape, desc0.shape, desc1.shape)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        desc0s, desc1s, all_matches = self.gnn(desc0, desc1)

        nI = len(desc0s)
        nB = desc0.shape[0]
        desc0s = torch.vstack(desc0s)  # [nI * nB, C, N]
        desc1s = torch.vstack(desc1s)
        mdescs = self.final_proj(torch.vstack([desc0s, desc1s]))
        dist = torch.einsum('bdn,bdm->bnm', mdescs[:nI * nB], mdescs[nI * nB:])
        dist = dist / self.config['descriptor_dim'] ** .5
        score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)  # [nI * nB, N, M]

        loss_out = self.match_net(score, data['matching_mask'].repeat(nI, 1, 1))

        if nI == 1:
            all_scores = [score]
        else:
            all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]
        loss_out['scores'] = all_scores
        loss = loss_out['matching_loss']  # currently no pose loss

        if not self.with_pose:
            loss_out['pose_loss'] = torch.zeros(size=[], device=desc0.device)

        loss_out['loss'] = loss

        return loss_out

    def forward_train_with_pose(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        # print(desc0.shape, desc1.shape, kpts0.shape, kpts1.shape, data['scores0'].shape, data['scores1'].shape)
        # print("norm: ", torch.sum(desc0 ** 2, dim=2)[0, 0:10])

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }
        # Keypoint normalization.
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        enc0 = self.kenc(norm_kpts0, data['scores0'])  # [B, C, N]
        enc1 = self.kenc(norm_kpts1, data['scores1'])
        # print('enc: ', enc0.shape, enc1.shape, desc0.shape, desc1.shape)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        desc0s, desc1s, all_matches = self.gnn(desc0, desc1)

        nI = len(desc0s)
        nB = desc0.shape[0]
        desc0s = torch.vstack(desc0s)  # [nI * nB, C, N]
        desc1s = torch.vstack(desc1s)
        mdescs = self.final_proj(torch.vstack([desc0s, desc1s]))
        dist = torch.einsum('bdn,bdm->bnm', mdescs[:nI * nB], mdescs[nI * nB:])
        dist = dist / self.config['descriptor_dim'] ** .5
        score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)  # [nI * nB, N, M]
        match_out = self.match_net(score, data['matching_mask'].repeat(nI, 1, 1))
        pose_out = self.pose_net(kpts0.repeat(nI, 1, 1), kpts1.repeat(nI, 1, 1), score[:, :-1, :-1],
                                 data['matching_mask'][:, :-1, :-1].repeat(nI, 1, 1))

        pose_loss = pose_out['pose_loss']
        valid_pose = pose_out['valid_pose']
        pred_pose = pose_out['pred_pose']
        inlier_ratio = pose_out['inlier_ratio']

        match_out['pose_loss'] = pose_loss
        loss = match_out['matching_loss'] + pose_loss  # currently no pose loss
        match_out['loss'] = loss
        match_out['valid_pose'] = valid_pose
        # print('valid_pose: ', valid_pose, pose_loss.item())

        if nI == 1:
            all_scores = [score]
            all_pred_poses = [pred_pose]
            all_inlier_ratios = [inlier_ratio]
        else:
            all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]
            all_pred_poses = [pred_pose[i * nB: (i + 1) * nB] for i in range(nI)]
            all_inlier_ratios = [inlier_ratio[i * nB: (i + 1) * nB] for i in range(nI)]

        match_out['scores'] = all_scores
        match_out['pred_pose'] = all_pred_poses
        match_out['inlier_ratio'] = all_inlier_ratios

        return match_out

    def produce_matches(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }

        # Keypoint normalization.
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        # Keypoint MLP encoder.
        enc0 = self.kenc(norm_kpts0, data['scores0'])  # [B, C, N]
        enc1 = self.kenc(norm_kpts1, data['scores1'])
        # print('enc: ', enc0.shape, enc1.shape, desc0.shape, desc1.shape)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # print('desc0: ', desc0[:, :5, :5])
        # print('desc1: ', desc1[:, :5, :5])

        # Multi-layer Transformer network.
        desc0s, desc1s, all_matches = self.gnn(desc0, desc1)

        nI = len(desc0s)
        nB = desc0.shape[0]
        desc0s = torch.vstack(desc0s)  # [nI * nB, C, N]
        desc1s = torch.vstack(desc1s)
        mdescs = self.final_proj(torch.vstack([desc0s, desc1s]))
        dist = torch.einsum('bdn,bdm->bnm', mdescs[:nI * nB], mdescs[nI * nB:])
        dist = dist / self.config['descriptor_dim'] ** .5
        score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)  # [nI * nB, N, M]
        indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=score)

        # compute correct matches
        if 'matching_mask' in data.keys():
            gt_matching_mask = data['matching_mask'].repeat(nI, 1, 1)
            gt_matches = torch.max(gt_matching_mask[:, :-1, :], dim=-1, keepdim=False)[1]
            acc_corr = torch.sum(((indices0 - gt_matches) == 0) * (indices0 != -1) * (
                    gt_matches < gt_matching_mask.shape[-1] - 1)) / (nB * nI)
            acc_incorr = torch.sum((indices0 == -1) * (gt_matches == gt_matching_mask.shape[-1] - 1)) / (nB * nI)
            acc_corr_total = torch.sum((gt_matches < gt_matching_mask.shape[-1] - 1)) / (nB * nI)
            acc_incorr_total = torch.sum((gt_matches == gt_matching_mask.shape[-1] - 1)) / (nB * nI)
        else:
            acc_corr = torch.zeros(size=[], device=desc0.device) + 0
            acc_corr_total = torch.zeros(size=[], device=desc0.device) + 1
            acc_incorr = torch.zeros(size=[], device=desc0.device) + 0
            acc_incorr_total = torch.zeros(size=[], device=desc0.device) + 1

        if nI == 1:
            all_scores = [score]
            all_indices0 = [indices0]
            all_mscores0 = [mscores0]
        else:
            all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]
            all_indices0 = [indices0[i * nB: (i + 1) * nB] for i in range(nI)]
            all_mscores0 = [mscores0[i * nB: (i + 1) * nB] for i in range(nI)]

        # print('score: ', all_scores[-1][:, :5, :5])
        output = {
            'scores': all_scores,
            'indices0': all_indices0,
            'mscores0': all_mscores0,
            'acc_corr': [acc_corr],
            'acc_incorr': [acc_incorr],
            'total_acc_corr': [acc_corr_total],
            'total_acc_incorr': [acc_incorr_total],
        }

        if self.with_pose:
            pose_out = self.pose_net(kpts0.repeat(nI, 1, 1), kpts1.repeat(nI, 1, 1), score[:, :-1, :-1],
                                     data['matching_mask'][:, :-1, :-1].repeat(nI, 1, 1))

            # pose_loss = pose_out['pose_loss']
            # valid_pose = pose_out['valid_pose']
            pred_pose = pose_out['pred_pose']
            inlier_ratio = pose_out['inlier_ratio']
            sampled_ids = pose_out['sampled_ids']

            if nI == 1:
                all_pred_poses = [pred_pose]
                all_inlier_ratios = [inlier_ratio]
                all_sampled_ids = [sampled_ids]
            else:
                all_pred_poses = [pred_pose[i * nB: (i + 1) * nB] for i in range(nI)]
                all_inlier_ratios = [inlier_ratio[i * nB: (i + 1) * nB] for i in range(nI)]
                all_sampled_ids = [sampled_ids[i * nB: (i + 1) * nB] for i in range(nI)]

            output['pred_pose'] = all_pred_poses
            output['inlier_ratio'] = all_inlier_ratios
            output['sampled_ids'] = all_sampled_ids

        return output

    def forward(self, data):
        if not self.training:
            return self.produce_matches(data=data)
        if self.with_pose:
            return self.forward_train_with_pose(data=data)
        else:
            return self.forward_train(data=data)

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        return self.gnn.forward_one_layer(desc0=desc0, desc1=desc1, M0=M0, M1=M1, layer_i=layer_i)

    def encode_keypoint(self, norm_kpts0, norm_kpts1, scores0, scores1):
        bs = norm_kpts0.shape[0]
        enc = self.kenc(torch.cat([norm_kpts0, norm_kpts1], dim=0), torch.cat([scores0, scores1], dim=0))  # [B, C, N]
        return enc[:bs], enc[bs:]

    def compute_distance(self, desc0, desc1, **kwargs):
        bs = desc0.shape[0]
        mdesc0 = self.final_proj(desc0)
        mdesc1 = self.final_proj(desc1)
        # mdesc = self.final_proj(torch.cat([desc0, desc1], dim=0))
        dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        dist = dist / self.config['descriptor_dim'] ** .5
        return dist

    def compute_score(self, dist):
        score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)  # [nB, N, M]

        return score

    def compute_matches_cycle(self, p):
        score, index = torch.topk(p, k=1, dim=-1)
        _, index2 = torch.topk(p, k=1, dim=-2)
        mask_th, index, index2 = score[:, 0] > self.config['match_threshold'], index[:, 0], index2.squeeze(0)
        mask_mc = index2[index] == torch.arange(len(p)).cuda()
        mask = mask_th & mask_mc
        index1, index2 = torch.nonzero(mask).squeeze(1), index[mask]
        return index1, index2, p[index1, index2]

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


class GraphMatcherP(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,  # [self, cross, self, cross, ...] 9 in total
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
        'with_pose': False,
        'with_label': False,
        'pooling_sizes': None,
        'with_hard_negative': False,
        'neg_margin': 0.2,
        'multi_scale': False,
        'multi_proj': False,

        'inlier_th': 0.25,
        'error_th': 10,
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**self.default_config, **config}
        print('Config in GraphMatcherP: ', self.config)

        self.neg_ratio = self.config['neg_ratio']
        self.sinkhorn_iterations = self.config['sinkhorn_iterations']
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.gnn = MultiClusterAttentionalGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pooling_sizes=self.config['pooling_sizes'],
        )
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'],
            self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.with_pose = self.config['with_pose']

        self.match_net = GraphLoss(config=self.config)
        # if self.with_pose:
        self.pose_net = PoseEstimator(config=self.config)

        self.layer_names = self.config['GNN_layers']

    def forward_train(self, data):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']
        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)

        bs = desc0.shape[0]
        M = desc0.shape[2]
        N = desc1.shape[2]

        # Keypoint normalization.
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        M01 = None
        M10 = None
        #
        M00 = torch.zeros(size=(bs, M, M), requires_grad=False).cuda()
        M11 = torch.zeros(size=(bs, N, N), requires_grad=False).cuda()

        pos_dist00 = kpts0.unsqueeze(3) - kpts0.unsqueeze(3).permute(0, 3, 2,
                                                                     1)  # [B, N, 2, 1] - [B, 1, 2, N] = [B, N, 2, N]
        pos_dist00 = torch.sum(pos_dist00 ** 2, dim=2)  # [B, M, M]
        dist00, indices00 = torch.topk(pos_dist00, largest=False, k=128, dim=2)
        M00 = M00.scatter(2, indices00, 1)
        pos_dist11 = kpts1.unsqueeze(3) - kpts1.unsqueeze(3).permute(0, 3, 2, 1)
        pos_dist11 = torch.sum(pos_dist11 ** 2, dim=2)  # N x N
        dist11, indices11 = torch.topk(pos_dist11, largest=False, k=128, dim=2)
        M11 = M11.scatter(2, indices11, 1)

        total_pose_loss = 0
        total_matching_loss = 0
        nI = len(self.layer_names) // 2

        output = {
            'scores': [],
            'pred_pose': [],
            'inlier_ratio': [],
            'matching_scores0': [],
        }

        for layer_i in range(nI):
            desc0, desc1 = self.forward_one_layer(desc0=desc0, desc1=desc1, M0=M00, M1=M11, layer_i=layer_i * 2)
            desc0, desc1 = self.forward_one_layer(desc0=desc0, desc1=desc1, M0=M01, M1=M10, layer_i=layer_i * 2 + 1)

            # TOTO: Update M
            dists = self.compute_distance(desc0=desc0, desc1=desc1)
            scores = self.compute_score(dist=dists)
            match_out = self.match_net(scores, data['matching_mask'])

            pose_out = self.pose_net(kpts0, kpts1, scores[:, :-1, :-1], data['matching_mask'][:, :-1, :-1])

            matching_loss = match_out['matching_loss']
            pose_loss = pose_out['pose_loss']

            # inlier_ratio = pose_out['inlier_ratio']
            support_connections = pose_out['support_connections']
            M01 = support_connections
            M10 = M01.transpose(1, 2)

            # print('M: ', torch.sum(M00) / (bs * M), torch.sum(M01) / (bs * M))

            total_pose_loss = total_pose_loss + pose_loss
            total_matching_loss = total_matching_loss + matching_loss

            output['scores'].append(scores)

            output['matching_scores0'] = match_out['matching_scores0']
            output['acc_corr'] = match_out['acc_corr']
            output['acc_incorr'] = match_out['acc_incorr']
            output['total_acc_corr'] = match_out['total_acc_corr']
            output['total_acc_incorr'] = match_out['total_acc_incorr']

            output['pred_pose'].append(pose_out['pred_pose'])
            output['inlier_ratio'].append(pose_out['inlier_ratio'])
            output['valid_pose'] = pose_out['valid_pose']

        total_pose_loss = total_pose_loss / nI
        total_matching_loss = total_matching_loss / nI

        output['loss'] = total_pose_loss + total_matching_loss
        output['matching_loss'] = total_matching_loss
        output['pose_loss'] = total_pose_loss

        return output

    def forward_eval(self, data):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']
        desc0 = desc0.transpose(1, 2)  # [B, D, N]
        desc1 = desc1.transpose(1, 2)

        bs = desc0.shape[0]
        M = desc0.shape[2]
        N = desc1.shape[2]

        # Keypoint normalization.
        if 'norm_keypoints0' in data.keys() and 'norm_keypoints1' in data.keys():
            norm_kpts0 = data['norm_keypoints0']
            norm_kpts1 = data['norm_keypoints1']
        elif 'image0' in data.keys() and 'image1' in data.keys():
            norm_kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            norm_kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        else:
            raise ValueError('Require image shape for keypoint coordinate normalization')

        enc0, enc1 = self.encode_keypoint(norm_kpts0=norm_kpts0, norm_kpts1=norm_kpts1, scores0=scores0,
                                          scores1=scores1)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        M01 = None
        M10 = None
        #
        M00 = torch.zeros(size=(bs, M, M), requires_grad=False).cuda()
        M11 = torch.zeros(size=(bs, N, N), requires_grad=False).cuda()

        pos_dist00 = kpts0.unsqueeze(3) - kpts0.unsqueeze(3).permute(0, 3, 2,
                                                                     1)  # [B, N, 2, 1] - [B, 1, 2, N] = [B, N, 2, N]
        pos_dist00 = torch.sum(pos_dist00 ** 2, dim=2)  # [B, M, M]
        dist00, indices00 = torch.topk(pos_dist00, largest=False, k=128, dim=2)
        M00 = M00.scatter(2, indices00, 1)
        pos_dist11 = kpts1.unsqueeze(3) - kpts1.unsqueeze(3).permute(0, 3, 2, 1)
        pos_dist11 = torch.sum(pos_dist11 ** 2, dim=2)  # N x N
        dist11, indices11 = torch.topk(pos_dist11, largest=False, k=128, dim=2)
        M11 = M11.scatter(2, indices11, 1)

        nI = len(self.layer_names) // 2

        output = {
            'scores': [],
            'pred_pose': [],
            'inlier_ratio': [],
            'valid_pose': [],
            'indices0': [],
            'mscores0': [],

        }

        for layer_i in range(nI):
            desc0, desc1 = self.forward_one_layer(desc0=desc0, desc1=desc1, M0=M00, M1=M11, layer_i=layer_i * 2)
            desc0, desc1 = self.forward_one_layer(desc0=desc0, desc1=desc1, M0=M01, M1=M10, layer_i=layer_i * 2 + 1)

            # TOTO: Update M
            dists = self.compute_distance(desc0=desc0, desc1=desc1)
            scores = self.compute_score(dist=dists)
            indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=scores)

            pose_out = self.pose_net(kpts0[0], kpts1[0], scores[0, :-1, :-1], None, indices0[0])

            # valid_pose = pose_out['valid_pose']
            # inlier_ratio = pose_out['inlier_ratio']
            support_connections = pose_out['support_connections']
            if len(support_connections.shape) == 2:
                support_connections = support_connections[None]
            M01 = support_connections
            M10 = M01.transpose(1, 2)

            # print('pose_out: ', valid_pose, inlier_ratio)

            output['indices0'].append(indices0)
            output['mscores0'].append(mscores0)

            output['scores'].append(scores)
        return output

    def forward(self, data):
        if self.training:
            return self.forward_train(data=data)
        else:
            return self.forward_eval(data=data)

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        return self.gnn.forward_one_layer(desc0=desc0, desc1=desc1, M0=M0, M1=M1, layer_i=layer_i)

    def encode_keypoint(self, norm_kpts0, norm_kpts1, scores0, scores1):
        bs = norm_kpts0.shape[0]
        enc = self.kenc(torch.cat([norm_kpts0, norm_kpts1], dim=0), torch.cat([scores0, scores1], dim=0))  # [B, C, N]
        # enc1 = self.kenc(norm_kpts1, scores1)
        return enc[0:bs], enc[bs:]

    def compute_distance(self, desc0, desc1):
        nb = desc0.shape[0]
        mdesc = self.final_proj(torch.cat([desc0, desc1], dim=0))
        dist = torch.einsum('bdn,bdm->bnm', mdesc[:nb], mdesc[nb:])
        dist = dist / self.config['descriptor_dim'] ** .5
        return dist

    def compute_score(self, dist):
        score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)  # [nB, N, M]
        return score

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


if __name__ == '__main__':
    config = {
        'superpoint': {
            'nms_radius': 3,
            'keypoint_threshold': 0.1,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': None,
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
            'descriptor_dim': 128,
            'neg_ratio': -1,
        }
    }

    nfeatures = 1024
    batch = 4
    M = 4
    kpts0 = torch.randint(0, 2, (batch, nfeatures, 2)).float().cuda()
    kpts1 = torch.randint(0, 2, (batch, nfeatures, 2)).float().cuda()
    scores0 = torch.rand((batch, nfeatures)).float().cuda()
    scores1 = torch.rand((batch, nfeatures)).float().cuda()
    descs0 = torch.rand((batch, nfeatures, 128)).float().cuda()
    descs1 = torch.rand((batch, nfeatures, 128)).float().cuda()
    descs0 = F.normalize(descs0, dim=1, p=2)
    descs1 = F.normalize(descs1, dim=1, p=2)
    labels0 = torch.randint(0, M, (batch, M, nfeatures)).long().cuda()
    labels1 = torch.randint(0, M, (batch, M, nfeatures)).long().cuda()

    for i in range(M):
        labels0[:, i, :] = i
        labels1[:, i, :] = i

    confidences0 = torch.rand((batch, M, nfeatures)).float().cuda()
    confidences1 = torch.rand((batch, M, nfeatures)).float().cuda()

    matching_mask = torch.randint(0, 1, (nfeatures + 1, nfeatures + 1)).float().cuda()
    model = GraphMatcher(config.get('superglue', {})).cuda()

    data = {
        'keypoints0': kpts0,
        'keypoints1': kpts1,
        'scores0': scores0,
        'scores1': scores1,
        'descriptors0': descs0,
        'descriptors1': descs1,
        'labels0': labels0,
        'labels1': labels1,
        'confidences0': confidences0,
        'confidences1': confidences1,
        'matching_mask': matching_mask,
    }
    for i in range(100):
        # with torch.no_grad():
        out = model(data)
        print(i, out.keys())

        # del out
