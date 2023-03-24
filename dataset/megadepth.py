# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp -> megadepth
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   30/11/2022 15:23
=================================================='''
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from dataset.utils import normalize_size_spg


def read_matches(scenes, base_path):
    print('Start loading matches...')
    matches = {}
    for scene in tqdm(scenes, total=len(scenes)):
        scene_pair_fn = os.path.join(base_path, 'training_data/matches_i3o5', scene + '.npy')
        if not os.path.isfile(scene_pair_fn):
            continue
        valid_pairs = np.load(scene_pair_fn, allow_pickle=True)
        matches[scene] = valid_pairs

    print('Loaded {:d} scenes of matches'.format(len(matches.keys())))

    return matches


class Megadepth(Dataset):
    def __init__(self,
                 scene_info_path,
                 base_path,
                 scene_list_fn,
                 pairs_per_scene=200,
                 image_size=256,
                 nfeatures=1024,
                 feature_type='spp',
                 train=True,
                 resize=-1,
                 tmp_dir=None,
                 min_inliers=32,
                 max_inliers=512,
                 random_inliers=False,
                 matches={},
                 **kwargs):
        self.scenes = []
        with open(scene_list_fn, 'r') as f:
            lines = f.readlines()
            for l in lines:
                self.scenes.append(l.strip())
        print('Load images from {:d} scenes'.format(len(self.scenes)))

        self.train = train
        self.scene_info_path = scene_info_path
        self.base_path = base_path
        self.scene_info = {}  # to be assigned at first loading if pre_load

        self.pairs_per_scene = pairs_per_scene
        self.image_size = image_size
        self.dataset = []
        self.min_inliers = min_inliers
        self.max_inliers = max_inliers
        self.random_inliers = random_inliers

        if resize == -1:
            self.resize = image_size
        else:
            self.resize = resize

        self.nfeatures = nfeatures
        self.feature_type = feature_type
        self.invalid_fns = []
        self.tmp_dir = tmp_dir
        self.keypoints = {}
        self.matches = matches
        if feature_type == 'spp':
            self.scene_nvalid_pairs = np.load('assets/mega_scene_nmatches_spp.npy', allow_pickle=True).item()
        else:
            self.scene_nvalid_pairs = np.load('assets/mega_scene_nmatches_sift.npy', allow_pickle=True).item()

        self.build_dataset(seed=0)

    def sample_matches_from_offline(self, idx):
        if not self.train:
            np.random.seed(0)
        pair = self.dataset[idx]
        scene_fn = pair[0]
        pair_id = pair[1]
        data = np.load(
            osp.join(self.base_path, 'training_data/matches_sep_{:s}'.format(self.feature_type), scene_fn,
                     "{:d}.npy".format(pair_id)),
            allow_pickle=True).item()
        '''
        data: dict = {'image_path1', 'depth_path1', 'intrinsics1', 'pose1', 'image_path2', 'depth_path2', 'intrinsics2', 'pose2', 'matches'}
        '''
        image_path1 = data['image_path1']
        image_path2 = data['image_path2']
        intrinsics1 = data['intrinsics1']
        intrinsics2 = data['intrinsics2']
        pose1 = data['pose1']
        pose2 = data['pose2']
        # matches = data['matches']

        # print('m1: ', len(matched_ids1) + len(unmatched_ids1), len(matched_ids2) + len(unmatched_ids2))

        scene = image_path1.split('/')[1]
        feat_fn1 = osp.join(self.base_path, 'training_data/keypoints', scene,
                            image_path1.split('/')[-1] + "_{:s}.npy".format(self.feature_type))
        feat_fn2 = osp.join(self.base_path, 'training_data/keypoints', scene,
                            image_path2.split('/')[-1] + "_{:s}.npy".format(self.feature_type))

        if feat_fn1 in self.invalid_fns or feat_fn2 in self.invalid_fns:
            return None

        if feat_fn1 in self.keypoints.keys():
            feat1 = self.keypoints[feat_fn1]
        else:
            feat1 = np.load(feat_fn1, allow_pickle=True).item()

        if feat_fn2 in self.keypoints.keys():
            feat2 = self.keypoints[feat_fn2]
        else:
            feat2 = np.load(feat_fn2, allow_pickle=True).item()

        kpts1 = feat1['keypoints']  # 4096 x 2
        scores1 = feat1['scores']
        descs1 = feat1['descriptors']
        image_size1 = feat1['image_size']  # [H, W, C]

        if kpts1.shape[0] < self.nfeatures:  # or len(matched_ids1) + len(unmatched_ids1) < self.nfeatures:
            self.invalid_fns.append(feat_fn1)
            print(feat_fn1)
            print('insufficient matches 1')
            return None

        kpts2 = feat2['keypoints']
        scores2 = feat2['scores']
        descs2 = feat2['descriptors']
        image_size2 = feat2['image_size']

        # print ('kpt2: ', kpts2.shape)

        if kpts2.shape[0] < self.nfeatures:  # or len(matched_ids2) + len(unmatched_ids2) < self.nfeatures:
            self.invalid_fns.append(feat_fn2)
            print(feat_fn2)
            print('insufficient matches 2')
            return None

        matched_ids1 = list(data['matched_ids1'])
        matched_ids2 = list(data['matched_ids2'])

        unmatched_ids1 = [i for i in range(kpts1.shape[0]) if i not in matched_ids1]
        unmatched_ids2 = [i for i in range(kpts2.shape[0]) if i not in matched_ids2]

        if len(matched_ids1) + len(unmatched_ids1) < self.nfeatures or len(matched_ids2) + len(
                unmatched_ids2) < self.nfeatures:
            return None

        n_matches = len(matched_ids1)
        n_left1 = kpts1.shape[0] - n_matches
        n_left2 = kpts2.shape[0] - n_matches

        matched_ids = [i for i in range(n_matches)]
        if self.train and self.random_inliers:
            n_inliers = np.random.randint(int(self.min_inliers), self.max_inliers + 1)
            if n_inliers < n_matches:
                n_matches = n_inliers

            n_matches_need = self.nfeatures - np.min([n_left1, n_left2])
            if n_matches_need > n_matches:
                n_matches = n_matches_need

            np.random.shuffle(matched_ids)
            matched_ids1 = np.array(matched_ids1)[matched_ids[:n_matches]].tolist()
            matched_ids2 = np.array(matched_ids2)[matched_ids[:n_matches]].tolist()
            # print(n_matches)

        if n_matches > self.nfeatures:
            sel_ids1 = matched_ids1[:self.nfeatures]
            sel_ids2 = matched_ids2[:self.nfeatures]
            n_matches = self.nfeatures
        else:
            np.random.shuffle(unmatched_ids1)
            np.random.shuffle(unmatched_ids2)
            sel_ids1 = matched_ids1 + unmatched_ids1[:self.nfeatures - n_matches]
            sel_ids2 = matched_ids2 + unmatched_ids2[:self.nfeatures - n_matches]

        matching_mask = np.zeros(shape=(self.nfeatures + 1, self.nfeatures + 1), dtype=float)
        shuffle_ids1 = np.arange(0, self.nfeatures)
        shuffle_ids2 = np.arange(0, self.nfeatures)
        np.random.shuffle(shuffle_ids1)
        np.random.shuffle(shuffle_ids2)

        sel_kpts1 = kpts1[sel_ids1][shuffle_ids1]
        sel_scores1 = scores1[sel_ids1][shuffle_ids1]
        sel_descs1 = descs1[sel_ids1][shuffle_ids1]

        sel_kpts2 = kpts2[sel_ids2][shuffle_ids2]
        sel_scores2 = scores2[sel_ids2][shuffle_ids2]
        sel_descs2 = descs2[sel_ids2][shuffle_ids2]

        for i in range(self.nfeatures):
            i1 = np.where(shuffle_ids1 == i)[0][0]
            i2 = np.where(shuffle_ids2 == i)[0][0]

            if i >= n_matches:
                matching_mask[i1, self.nfeatures] = 1
                matching_mask[self.nfeatures, i2] = 1
            else:
                matching_mask[i1, i2] = 1

        P21 = pose2 @ np.linalg.inv(pose1)  # [4, 4]
        t0, t1, t2 = P21[:3, 3]  # / np.linalg.norm(P21[:3, 3], ord=2, keepdims=True)
        t_skew = np.array([
            [0, -t2, t1],
            [t2, 0, -t0],
            [-t1, t0, 0]
        ])
        E21 = t_skew @ P21[:3, :3]
        F21 = np.linalg.inv(intrinsics2.transpose()) @ E21[0:3, 0:3] @ np.linalg.inv(intrinsics1)

        sel_kpts1_3d = (sel_kpts1 - intrinsics1[[0, 1], [2, 2]][None]) / intrinsics1[[0, 1], [0, 1]][None]
        sel_kpts2_3d = (sel_kpts2 - intrinsics2[[0, 1], [2, 2]][None]) / intrinsics2[[0, 1], [0, 1]][None]

        return {
            'keypoints0': sel_kpts1,
            'keypoints1': sel_kpts2,

            'keypoints0_3d': sel_kpts1_3d,
            'keypoints1_3d': sel_kpts2_3d,

            'norm_keypoints0': normalize_size_spg(x=sel_kpts1, size=np.array([image_size1[1], image_size1[0]], int)),
            'norm_keypoints1': normalize_size_spg(x=sel_kpts2, size=np.array([image_size2[1], image_size2[0]], int)),

            'descriptors0': sel_descs1,
            'descriptors1': sel_descs2,

            'scores0': sel_scores1,
            'scores1': sel_scores2,

            'intrinsics0': intrinsics1,
            'intrinsics1': intrinsics2,

            'matching_mask': matching_mask,
            'gt_pose': F21,
            'P21': P21,
            'gt_E': E21,

            'file_name': '{:s}_{:s}_{:s}'.format(scene, image_path1.split('/')[-1], image_path2.split('/')[-1]),
            # 'file_name': [image_path1, image_path2],

            'image_size0': np.array([image_size1[2], image_size1[0], image_size1[1]], dtype=int).reshape(1, 3),
            'image_size1': np.array([image_size2[2], image_size2[0], image_size2[1]], dtype=int).reshape(1, 3),
            'file_name0': image_path1,
            'file_name1': image_path2,
        }

    def __getitem__(self, idx):
        for i in range(len(self.dataset)):
            ni = (i + idx) % len(self.dataset)
            output = self.sample_matches_from_offline(idx=ni)
            if output is not None:
                break
        return output

    def __len__(self):
        return len(self.dataset)

    def build_dataset(self, seed=-1):
        self.build_dataset_from_offline(seed=seed)

    def build_dataset_from_offline(self, seed=-1):
        self.dataset = []
        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            # np.random.seed(time.time())
            print('Building the validation dataset...')
        else:
            if seed >= 0:
                np.random.seed(seed=seed)
            print('Building a new training dataset...')

        # scene_pair_file = h5py.File(osp.join(self.base_path, 'matches_spp.h5'), 'r')

        for scene in tqdm(self.scenes, total=len(self.scenes)):
            if scene not in self.scene_nvalid_pairs.keys():
                continue
            valid_pairs = [i for i in range(self.scene_nvalid_pairs[scene])]

            if len(valid_pairs) <= self.pairs_per_scene:
                selected_ids = np.arange(0, len(valid_pairs))
            else:
                try:
                    selected_ids = np.random.choice(
                        len(valid_pairs), self.pairs_per_scene
                    )
                except:
                    continue
            for sid in selected_ids:
                self.dataset.append((scene, sid))

        # print('rank: ', self.rank, self.dataset[:10])
        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

        print('Loaded {:d} keypoints and {:d} matches'.format(len(self.keypoints.keys()), len(self.matches.keys())))
