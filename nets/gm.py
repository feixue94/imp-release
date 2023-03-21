from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from nets.loss import GraphLoss
from nets.poses import PoseEstimator, PoseEstimatorV3

from nets.basic_layers import sink_algorithm
from nets.basic_layers import dual_softmax


class SampleLN(nn.Module):
    def __init__(self, dim, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        if len(x.shape) == 2:  # [B, D]
            return self.norm(x)
        elif len(x.shape) == 3:  # [B, D, N]
            b = x.shape[0]
            y = self.norm(rearrange(x, 'b d n -> (b n) d'))
            return rearrange(y, '(b n) d -> b d n', b=b)


def MLP(channels: list, do_bn=True, ac_fn='relu', norm_fn='bn'):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        if norm_fn == 'ln':
            if i == 1:
                layers.append(SampleLN(channels[i - 1]))
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if norm_fn == 'in':
                layers.append(nn.InstanceNorm1d(channels[i], eps=1e-3))
            elif norm_fn == 'bn':
                layers.append(nn.BatchNorm1d(channels[i], eps=1e-3))

            if ac_fn == 'relu':
                layers.append(nn.ReLU())
            elif ac_fn == 'gelu':
                layers.append(nn.GELU())
            elif ac_fn == 'lrelu':
                layers.append(nn.LeakyReLU(negative_slope=0.1))
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2 + 0.5
    scaling = size.max(1, keepdim=True).values * 0.7
    norm_kpts = (kpts - center[:, None, :]) / scaling[:, None, :]
    # print(center, size, image_shape)
    # print(scaling)
    # print('kpts: ', kpts)
    # print('size: ', image_shape)
    # print('norm_kpts: ', norm_kpts)
    # exit(0)
    return norm_kpts


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers, ac_fn='relu', norm_fn='bn'):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim], ac_fn=ac_fn, norm_fn=norm_fn)
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
        self.prob = prob

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
    def __init__(self, feature_dim: int, num_heads: int, ac_fn: str = 'relu', norm_fn: str = 'bn'):
        super().__init__()
        self.attn = MultiClusterMultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim], ac_fn=ac_fn, norm_fn=norm_fn)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward_label(self, x, source, labels_x=None, labels_s=None, confidences_x=None, confidences_s=None):
        message = self.attn(x, source, source, labels_x, labels_s, confidences_x, confidences_s)
        self.prob = self.attn.prob
        return self.mlp(torch.cat([x, message], dim=1))

    def forward(self, x, source, M=None):
        message = self.attn(x, source, source, M=M)
        self.prob = self.attn.prob
        return self.mlp(torch.cat([x, message], dim=1))


class MultiClusterAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, pooling_sizes: list = None, keep_ratios: list = None,
                 ac_fn: str = 'relu', norm_fn: str = 'bn'):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiClusterAttentionalPropagation(feature_dim, 4, ac_fn=ac_fn, norm_fn=norm_fn)
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
                # delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc1, desc0], dim=0), None)
                delta0 = layer(desc0, desc1, None)
                delta1 = layer(desc1, desc0, None)
            else:
                # delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc0, desc1], dim=0), None)
                delta0 = layer(desc0, desc0, None)
                delta1 = layer(desc1, desc1, None)
                # delta0 = delta[:bs]
                # delta1 = delta[bs:]

            desc0 = (desc0 + delta0)
            desc1 = (desc1 + delta1)

            if name == 'cross':
                all_desc0s.append(desc0)
                all_desc1s.append(desc1)
        return all_desc0s, all_desc1s, []

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        bs = desc0.shape[0]
        layer = self.layers[layer_i]
        name = self.names[layer_i]
        if name == 'cross':
            # delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc1, desc0], dim=0),
            #               None if M0 is None or M1 is None else torch.cat([M0, M1], dim=0))
            # delta0 = delta[:bs]
            # delta1 = delta[bs:]
            delta0 = layer(desc0, desc1, None)
            delta1 = layer(desc1, desc0, None)
        else:
            # delta = layer(torch.cat([desc0, desc1], dim=0), torch.cat([desc0, desc1], dim=0),
            #               None if M0 is None or M1 is None else torch.cat([M0, M1], dim=0))
            # delta0 = delta[:bs]
            # delta1 = delta[bs:]
            delta0 = layer(desc0, desc0, None)
            delta1 = layer(desc1, desc1, None)

        return desc0 + delta0, desc1 + delta1

    def update_connections_by_pooling_size(self, last_connections, last_prob, pool_size):
        if last_connections is None:
            return None
        if last_prob is None or pool_size is None:
            return last_connections
        if pool_size == 1:
            return last_connections

        with torch.no_grad():
            current_connections = torch.zeros_like(last_connections)
            k = int(torch.sum(last_connections[0], dim=-1)[0].item() // pool_size)
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


class GM(nn.Module):
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
        'n_layers': 9,
        'n_min_tokens': -1,
        'with_sinkhorn': True,

        'ac_fn': 'relu',
        'norm_fn': 'bn',
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        print('config in GM: ', self.config)

        self.neg_ratio = self.config['neg_ratio']
        self.multi_scale = self.config['multi_scale']
        self.multi_proj = self.config['multi_proj']
        self.n_layers = self.config['n_layers']

        self.with_sinkhorn = self.config['with_sinkhorn']
        self.match_threshold = self.config['match_threshold']

        self.sinkhorn_iterations = self.config['sinkhorn_iterations']
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'])
        self.gnn = MultiClusterAttentionalGNN(
            feature_dim=self.config['descriptor_dim'],
            layer_names=self.config['GNN_layers'],
            pooling_sizes=self.config['pooling_sizes'],
            ac_fn=self.config['ac_fn'],
            norm_fn=self.config['norm_fn'],
        )

        if self.multi_proj:
            self.final_proj = nn.ModuleList([nn.Conv1d(
                self.config['descriptor_dim'],
                self.config['descriptor_dim'],
                kernel_size=1, bias=True) for _ in range(self.n_layers)])
        else:
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
            # self.pose_net = PoseEstimator(config=self.config)
            self.pose_net = PoseEstimatorV3(config=self.config)

        self.self_prob0 = None
        self.self_prob1 = None
        self.cross_prob0 = None
        self.cross_prob1 = None

    def forward_train(self, data, p=0.2):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

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
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        desc0s, desc1s, all_matches = self.gnn(desc0, desc1)

        nI = len(desc0s)
        nB = desc0.shape[0]

        if not self.multi_proj:
            desc0s = torch.vstack(desc0s)  # [nI * nB, C, N]
            desc1s = torch.vstack(desc1s)
            mdescs = self.final_proj(torch.vstack([desc0s, desc1s]))
        else:
            mdescs0 = []
            mdescs1 = []
            for l, d0, d1 in zip(self.final_proj, desc0s, desc1s):
                md = l(torch.vstack([d0, d1]))
                mdescs0.append(md[:nB])
                mdescs1.append(md[nB:])
            mdescs = torch.vstack([torch.vstack(mdescs0), torch.vstack(mdescs1)])

        dist = torch.einsum('bdn,bdm->bnm', mdescs[:nI * nB], mdescs[nI * nB:])
        dist = dist / self.config['descriptor_dim'] ** .5
        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

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
        scores0, scores1 = data['scores0'], data['scores1']

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
        if not self.multi_proj:
            desc0s = torch.vstack(desc0s)  # [nI * nB, C, N]
            desc1s = torch.vstack(desc1s)
            mdescs = self.final_proj(torch.vstack([desc0s, desc1s]))
        else:
            mdescs0 = []
            mdescs1 = []
            for f, d0, d1 in zip(self.final_proj, desc0s, desc1s):
                md = f(torch.vstack([d0, d1]))
                mdescs0.append(md[:nB])
                mdescs1.append(md[nB:])
            mdescs = torch.vstack([torch.vstack(mdescs0), torch.vstack(mdescs1)])
        dist = torch.einsum('bdn,bdm->bnm', mdescs[:nI * nB], mdescs[nI * nB:])
        dist = dist / self.config['descriptor_dim'] ** .5
        # score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)  # [nI * nB, N, M]
        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

        match_out = self.match_net(score, data['matching_mask'].repeat(nI, 1, 1))
        pose_out = self.pose_net(kpts0.repeat(nI, 1, 1), kpts1.repeat(nI, 1, 1), score[:, :-1, :-1],
                                 data['matching_mask'][:, :-1, :-1].repeat(nI, 1, 1))

        pose_loss = pose_out['pose_loss'] + pose_out['geo_loss']
        valid_pose = pose_out['valid_pose']
        pred_pose = pose_out['pred_pose']
        inlier_ratio = pose_out['inlier_ratio']

        match_out['pose_loss'] = pose_loss
        loss = match_out['matching_loss'] + pose_loss  # currently no pose loss
        match_out['loss'] = loss
        match_out['valid_pose'] = valid_pose

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

    def forward_train_with_pose_v2(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

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
        if not self.multi_proj:
            desc0s = torch.vstack(desc0s)  # [nI * nB, C, N]
            desc1s = torch.vstack(desc1s)
            mdescs = self.final_proj(torch.vstack([desc0s, desc1s]))
        else:
            mdescs0 = []
            mdescs1 = []
            for f, d0, d1 in zip(self.final_proj, desc0s, desc1s):
                md = f(torch.vstack([d0, d1]))
                mdescs0.append(md[:nB])
                mdescs1.append(md[nB:])
            mdescs = torch.vstack([torch.vstack(mdescs0), torch.vstack(mdescs1)])
        dist = torch.einsum('bdn,bdm->bnm', mdescs[:nI * nB], mdescs[nI * nB:])
        dist = dist / self.config['descriptor_dim'] ** .5
        # score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)  # [nI * nB, N, M]
        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

        match_out = self.match_net(score, data['matching_mask'].repeat(nI, 1, 1))

        # pred_indices0, pred_indices1, pred_mscores0, pred_mscores1 = self.compute_matches(scores=score, p=self.config[
        #     'match_threshold'])
        # pose_out = self.pose_net(kpts0.repeat(nI, 1, 1), kpts1.repeat(nI, 1, 1), score[:, :-1, :-1],
        #                          data['matching_mask'][:, :-1, :-1].repeat(nI, 1, 1))
        pred_indices0 = match_out['matches0']
        pred_mscores0 = match_out['matching_scores0']
        # kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        # kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
        # K0s = data['intrinsics0']
        # K1s = data['intrinsics1']
        e_gt = torch.reshape(data['gt_E'], (-1, 9))
        # e_gt = torch.reshape(data['gt_pose'], (-1, 9))
        # e_gt = e_gt_unnorm / torch.norm(e_gt_unnorm, dim=1, keepdim=True)
        K0s = data['intrinsics0']
        K1s = data['intrinsics1']
        pose_out = self.pose_net(
            data['keypoints0_3d'].repeat(nI, 1, 1),
            data['keypoints1_3d'].repeat(nI, 1, 1),
            # kpts0.repeat(nI, 1, 1),
            # kpts1.repeat(nI, 1, 1),
            pred_indices0,
            pred_mscores0,
            e_gt.repeat(nI, 1),
            data['matching_mask'][:, :-1, :-1].repeat(nI, 1, 1),
            fs=(K0s[:, 0, 0] + K0s[:, 1, 1] + K1s[:, 0, 0] + K1s[:, 1, 1]).repeat(nI, 1) / 4,
        )

        pose_loss = pose_out['pose_loss']
        geo_loss = pose_out['geo_loss']

        match_out['pose_loss'] = pose_loss + geo_loss
        loss = match_out['matching_loss'] + (pose_loss + geo_loss)  # * 0.1
        match_out['loss'] = loss
        # print('loss: ', loss.item(), match_out['matching_loss'].item(), pose_loss.item(), geo_loss.item())
        # match_out['valid_pose'] = 0
        # print('valid_pose: ', valid_pose, pose_loss.item())

        if nI == 1:
            all_scores = [score]
            # all_pred_poses = [pred_pose]
            # all_inlier_ratios = [inlier_ratio]
        else:
            all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]
            # all_pred_poses = [pred_pose[i * nB: (i + 1) * nB] for i in range(nI)]
            # all_inlier_ratios = [inlier_ratio[i * nB: (i + 1) * nB] for i in range(nI)]

        match_out['scores'] = all_scores
        # match_out['pred_pose'] = all_pred_poses
        # match_out['inlier_ratio'] = all_inlier_ratios

        return match_out

    def produce_matches(self, data, p=0.2, only_last=False, **kwargs):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']

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
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        desc0s, desc1s, all_matches = self.gnn(desc0, desc1)

        nI = len(desc0s)
        nB = desc0.shape[0]

        if not self.multi_proj:
            if only_last:
                mdescs0 = self.final_proj(desc0s[-1])
                mdescs1 = self.final_proj(desc1s[-1])
            else:
                desc0s = torch.vstack(desc0s)  # [nI * nB, C, N]
                desc1s = torch.vstack(desc1s)

                mdescs0 = self.final_proj(desc0s)
                mdescs1 = self.final_proj(desc1s)
        else:
            if only_last:
                mdescs0 = self.final_proj[-1](desc0s[-1])
                mdescs1 = self.final_proj[-1](desc1s[-1])
            else:
                mdescs0 = []
                mdescs1 = []
                for l, d0, d1 in zip(self.final_proj, desc0s, desc1s):
                    # md = l(torch.vstack([d0, d1]))
                    md0 = l(d0)
                    md1 = l(d1)
                    mdescs0.append(md0)
                    mdescs1.append(md1)

                mdescs0 = torch.vstack(mdescs0)
                mdescs1 = torch.vstack(mdescs1)

        dist = torch.einsum('bdn,bdm->bnm', mdescs0, mdescs1)
        dist = dist / self.config['descriptor_dim'] ** .5
        score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)  # [nI * nB, N, M]
        # score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

        indices0, indices1, mscores0, mscores1 = self.compute_matches(scores=score, p=p)

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
            if only_last:
                all_indices0 = [indices0]
                all_indices1 = [indices1]
                all_mscores0 = [mscores0]
                all_mscores1 = [mscores1]
                all_scores = [score]
            else:
                all_scores = [score[i * nB: (i + 1) * nB] for i in range(nI)]
                all_indices0 = [indices0[i * nB: (i + 1) * nB] for i in range(nI)]
                all_mscores0 = [mscores0[i * nB: (i + 1) * nB] for i in range(nI)]

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

    def produce_matches_test(self, data, p=0.2, only_last=False, **kwargs):
        return self.produce_matches(data=data, p=p, only_last=only_last, kwargs=kwargs)

    def forward(self, data, mode=0):
        if not self.training:
            if mode == 0:
                return self.produce_matches(data=data)
            else:
                return self.run(data=data)
        if self.with_pose:
            # return self.forward_train_with_pose(data=data)
            return self.forward_train_with_pose_v2(data=data)
        else:
            return self.forward_train(data=data)

    def forward_one_layer_old(self, desc0, desc1, M0, M1, layer_i):
        return self.gnn.forward_one_layer(desc0=desc0, desc1=desc1, M0=M0, M1=M1, layer_i=layer_i)

    def forward_one_layer(self, desc0, desc1, M0, M1, layer_i):
        layer = self.gnn.layers[layer_i]
        name = self.gnn.names[layer_i]

        if name == 'cross':
            ds_desc0 = desc0
            ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc1, M=None)
            self.cross_prob1 = layer.prob
            delta1 = layer(desc1, ds_desc0, M=None)
            self.cross_prob0 = layer.prob

        elif name == 'self':
            ds_desc0 = desc0
            ds_desc1 = desc1

            delta0 = layer(desc0, ds_desc0, M=None)
            self.self_prob0 = layer.prob
            delta1 = layer(desc1, ds_desc1, M=None)
            self.self_prob1 = layer.prob

        return desc0 + delta0, desc1 + delta1

    def encode_keypoint(self, norm_kpts0, norm_kpts1, scores0, scores1):
        return self.kenc(norm_kpts0, scores0), self.kenc(norm_kpts1, scores1)

    def compute_distance(self, desc0, desc1, layer_id=-1):
        if not self.multi_proj:
            mdesc0 = self.final_proj(desc0)
            mdesc1 = self.final_proj(desc1)
        else:
            mdesc0 = self.final_proj[layer_id](desc0)
            mdesc1 = self.final_proj[layer_id](desc1)
        dist = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        dist = dist / self.config['descriptor_dim'] ** .5
        return dist

    def compute_score(self, dist, dustbin, iteration):
        # score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)  # [nB, N, M]
        if self.with_sinkhorn:
            score = sink_algorithm(M=dist, dustbin=dustbin,
                                   iteration=iteration)  # [nI * nB, N, M]
        else:
            score = dual_softmax(M=dist, dustbin=dustbin)
        return score

    def compute_score_sinkhorn(self, dist):
        score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)
        return score

    def compute_matches_cycle(self, p):
        score, index = torch.topk(p, k=1, dim=-1)
        _, index2 = torch.topk(p, k=1, dim=-2)
        mask_th, index, index2 = score[:, 0] > self.config['match_threshold'], index[:, 0], index2.squeeze(0)
        mask_mc = index2[index] == torch.arange(len(p)).cuda()
        mask = mask_th & mask_mc
        index1, index2 = torch.nonzero(mask).squeeze(1), index[mask]
        return index1, index2, p[index1, index2]

    def compute_matches(self, scores, p=0.2):
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores0 = torch.where(mutual0, max0.values, zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        # valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid0 = mutual0 & (mscores0 > p)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return indices0, indices1, mscores0, mscores1

    def run(self, data):
        desc0 = data['desc1']
        # print('desc0: ', torch.sum(desc0 ** 2, dim=-1))
        # desc0 = torch.nn.functional.normalize(desc0, dim=-1)
        desc0 = desc0.transpose(1, 2)

        desc1 = data['desc2']
        # desc1 = torch.nn.functional.normalize(desc1, dim=-1)
        desc1 = desc1.transpose(1, 2)

        kpts0 = data['x1'][:, :, :2]
        kpts1 = data['x2'][:, :, :2]
        # kpts0 = normalize_keypoints(kpts=kpts0, image_shape=data['image_shape1'])
        # kpts1 = normalize_keypoints(kpts=kpts1, image_shape=data['image_shape2'])
        scores0 = data['x1'][:, :, -1]
        scores1 = data['x2'][:, :, -1]

        # Keypoint MLP encoder.
        enc0 = self.kenc(kpts0, scores0)  # [B, C, N]
        enc1 = self.kenc(kpts1, scores1)
        # print('enc: ', enc0.shape, enc1.shape, desc0.shape, desc1.shape)
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        desc0s, desc1s, all_matches = self.gnn(desc0, desc1)

        nI = len(desc0s)
        nB = desc0.shape[0]
        # print(nI, nB)
        desc0s = desc0s[-1]  # [nI * nB, C, N]
        desc1s = desc1s[-1]
        if self.multi_proj:
            mdescs0 = self.final_proj[-1](desc0s)
            mdescs1 = self.final_proj[-1](desc1s)
        else:
            mdescs0 = self.final_proj(desc0s)
            mdescs1 = self.final_proj(desc1s)

        dist = torch.einsum('bdn,bdm->bnm', mdescs0, mdescs1)
        dist = dist / self.config['descriptor_dim'] ** .5
        # score = sink_algorithm(M=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)  # [nI * nB, N, M]
        score = self.compute_score(dist=dist, dustbin=self.bin_score, iteration=self.sinkhorn_iterations)

        # print('score: ', score)
        output = {
            'p': score,
        }

        return output


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
            'multi_scale': True,
            'multi_proj': True,
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

    matching_mask = torch.randint(0, 1, (nfeatures + 1, nfeatures + 1)).float().cuda()
    matching_mask = matching_mask[None].repeat(batch, 1, 1)
    model = GM(config.get('superglue', {})).cuda().eval()
    print('model: ', model)

    data = {
        'keypoints0': kpts0,
        'keypoints1': kpts1,
        'norm_keypoints0': kpts0,
        'norm_keypoints1': kpts1,
        'scores0': scores0,
        'scores1': scores1,
        'descriptors0': descs0,
        'descriptors1': descs1,
        'matching_mask': matching_mask,
    }
    for i in range(100):
        with torch.no_grad():
            out = model(data)
        print(i, out.keys())
        exit(0)

        # del out
