# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp -> dump_megadepth
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   30/11/2022 12:01
=================================================='''
import os
import os.path as osp
import numpy as np
import cv2
import torch
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset
from nets.superpoint import SuperPoint
from tools.geometry import match_from_projection_points_torch


def plot_matches_cv2(image0, image1, kpts0, kpts1, matches, margin=10, inliers=None):
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]

    h = max(h0, h1)
    w = w0 + w1 + margin
    if len(image0.shape) == 2:
        match_img = np.zeros((h, w), np.uint8)
    else:
        match_img = np.zeros((h, w, 3), np.uint8)
    match_img[:h0, :w0] = image0
    match_img[:h1, w0 + margin:] = image1
    if len(match_img.shape) == 2:
        match_img = cv2.cvtColor(match_img, cv2.COLOR_GRAY2BGR)

    for i in range(kpts0.shape[0]):
        pt = kpts0[i]
        match_img = cv2.circle(match_img, center=(int(pt[0]), int(pt[1])), radius=3, thickness=1, color=(0, 0, 255))

    for i in range(kpts1.shape[0]):
        pt = kpts1[i]
        match_img = cv2.circle(match_img, center=(int(pt[0] + w0 + margin), int(pt[1])), radius=3, thickness=1,
                               color=(0, 0, 255))

    for i in range(matches.shape[0]):
        p0 = kpts0[matches[i, 0]]
        p1 = kpts1[matches[i, 1]]
        match_img = cv2.line(match_img, pt1=(int(p0[0]), int(p0[1])), pt2=(int(p1[0] + w0 + margin), int(p1[1])),
                             color=(0, 255, 0), thickness=2)
    return match_img


class Megadepth:
    def __init__(self, scene_info_path,
                 base_path,
                 scene_list_fn,
                 min_overlap_ratio=0.1,
                 max_overlap_ratio=0.7,
                 max_scale_ratio=np.inf,
                 nfeatures=4096,
                 feature_type='sift',
                 inlier_th=5,
                 extract_features=True,
                 ):

        self.scene_info_path = scene_info_path
        self.base_path = base_path
        self.scene_list_fn = scene_list_fn
        self.feature_type = feature_type
        self.nfeatures = nfeatures
        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_scale_ratio = max_scale_ratio
        self.inlier_th = inlier_th

        self.scenes = []
        with open(scene_list_fn, 'r') as f:
            lines = f.readlines()
            for l in lines:
                self.scenes.append(l.strip())
        print('Load images from {:d} scenes'.format(len(self.scenes)))

        if extract_features:
            if feature_type == 'sift':
                self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures,
                                                        contrastThreshold=0.04)
            elif feature_type == 'spp':
                spp_config = {
                    'descriptor_dim': 256,
                    'nms_radius': 3,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': self.nfeatures,
                    'remove_borders': 4,
                    'weight_path': '/home/mifs/fx221/Research/Code/pnba/weights/superpoint_v1.pth',
                    'with_compensate': True,
                }
                self.spp = SuperPoint(config=spp_config).eval().cuda()

        self.image_paths = []
        self.depth_paths = []
        self.poses = []
        self.intrinsics = []

        self.extract_image_fns()

    def detectAndCompute(self, img):
        if self.feature_type == 'sift':
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            # kpts, descs = self.sift.detectAndCompute(img_gray, None)
            #
            # # print('descs: ', descs.shape, np.sqrt(np.sum((descs / 512.) ** 2, axis=1)))
            #
            # scores = np.array([kp.response for kp in kpts]).reshape(-1)
            # kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kpts])
            # descs = (descs / 512.).astype(float)

            cv_kp, desc = self.sift.detectAndCompute(img_gray, None)
            kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.response] for _kp in cv_kp])  # N*3
            index = np.flip(np.argsort(kp[:, 2]))
            kp, desc = kp[index], desc[index]
            descs = np.sqrt(abs(desc / (np.linalg.norm(desc, axis=-1, ord=1)[:, np.newaxis] + 1e-8)))

            kps = kp[:, 0:2]
            scores = kp[:, 2].reshape(-1, )

        elif self.feature_type == 'spp':
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            norm_img = img_gray.astype(float) / 255.
            with torch.no_grad():
                # print('norm_img: ', img_gray.shape, norm_img.shape)
                norm_img = torch.from_numpy(norm_img[None, None]).cuda().float()
                outputs = self.spp({'image': norm_img})

                kps = torch.vstack(outputs['keypoints']).cpu().numpy()
                descs = torch.vstack(outputs['descriptors']).cpu().numpy().transpose()
                scores = torch.vstack(outputs['scores']).cpu().numpy().reshape(-1, )
                del norm_img
                del outputs

        return kps, scores, descs

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]
        pose = self.poses[idx]
        intrinsic = self.intrinsics[idx]

        with h5py.File(osp.join(self.base_path, depth_path), 'r') as hf5_f:
            depth = np.array(hf5_f['/depth'])
        assert (np.min(depth) >= 0)

        image = cv2.imread(osp.join(self.base_path, image_path))
        kpts, scores, descs = self.detectAndCompute(img=image)
        depth_values = depth[kpts.astype(int)[:, 1], kpts.astype(int)[:, 0]]

        out = {
            'image_path': image_path,
            'depth_path': depth_path,
            'keypoints': kpts,
            'scores': scores,
            'descriptors': descs,
            'image_size': np.array(image.shape, int),
            'depth': depth_values,
            'pose': pose,
            'intrinsics': intrinsic,
        }

        return out

    def __len__(self):
        return len(self.image_paths)

    def build_correspondence(self, scene, save_dir, show_match=False, pre_load=False):
        keypoint_dir = osp.join(save_dir, 'keypoints_{:s}'.format(self.feature_type), scene)
        match_dir = osp.join(save_dir, 'matches_{:s}'.format(self.feature_type))
        print(keypoint_dir, match_dir)

        if osp.isfile(osp.join(match_dir, scene + '.npy')):
            return

        scene_info_path = osp.join(self.scene_info_path, '{:s}.0.npz'.format(scene))
        print(scene_info_path)

        if not osp.exists(scene_info_path):
            print('{:s} does not exit'.format(scene_info_path))
            return

        if osp.exists(osp.join(match_dir, scene + '.npy')):
            print('{:s} exist'.format(scene))
            return

        scene_info = np.load(scene_info_path, allow_pickle=True)
        overlap_matrix = scene_info['overlap_matrix']
        scale_ratio_matrix = scene_info['scale_ratio_matrix']

        valid = np.logical_and(
            np.logical_and(overlap_matrix >= self.min_overlap_ratio,
                           overlap_matrix <= self.max_overlap_ratio),
            scale_ratio_matrix <= self.max_scale_ratio
        )

        pairs = np.vstack(np.where(valid))
        selected_ids = np.arange(0, pairs.shape[1])
        print('Find {:d} pairs from scene {:s}'.format(len(selected_ids), scene))

        image_paths = scene_info['image_paths']
        depth_paths = scene_info['depth_paths']
        points3D_id_to_2D = scene_info['points3D_id_to_2D']
        # points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
        intrinsics = scene_info['intrinsics']
        poses = scene_info['poses']
        valid_pairs = []

        all_keypoints = {}
        if pre_load:
            print('Loading keypoints...')
            for img_path in tqdm(image_paths, total=len(image_paths)):
                if img_path is None:
                    continue
                kpt_fn = osp.join(keypoint_dir, img_path.split('/')[-1] + '_' + feature_type + '.npy')
                if osp.isfile(kpt_fn):
                    data = np.load(kpt_fn, allow_pickle=True).item()
                    all_keypoints[kpt_fn] = data

        if show_match:
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)

        for pair_idx in tqdm(selected_ids, total=len(selected_ids)):
            idx1 = pairs[0, pair_idx]
            idx2 = pairs[1, pair_idx]

            matches = np.array(list(
                points3D_id_to_2D[idx1].keys() &
                points3D_id_to_2D[idx2].keys()
            ))

            if len(matches) < 20:
                continue

            # conditions for rejecting pairs
            # number of spp points
            # image size
            image_path1 = image_paths[idx1]
            image_path2 = image_paths[idx2]
            # depth_path1 = depth_paths[idx1]
            # depth_path2 = depth_paths[idx2]
            pose1 = poses[idx1]
            pose2 = poses[idx2]
            intrinsics1 = intrinsics[idx1]
            intrinsics2 = intrinsics[idx2]

            kpt_fn1 = osp.join(keypoint_dir, image_path1.split('/')[-1] + '_' + feature_type + '.npy')
            if kpt_fn1 in all_keypoints.keys():
                data1 = all_keypoints[kpt_fn1]
            else:
                data1 = np.load(kpt_fn1, allow_pickle=True).item()
                all_keypoints[kpt_fn1] = data1

            kpts1 = data1['keypoints']
            depth1 = data1['depth']

            if kpts1.shape[0] < 1024:
                continue

            kpt_fn2 = osp.join(keypoint_dir, image_path2.split('/')[-1] + '_' + feature_type + '.npy')
            if kpt_fn2 in all_keypoints.keys():
                data2 = all_keypoints[kpt_fn2]
            else:
                data2 = np.load(kpt_fn2, allow_pickle=True).item()
                all_keypoints[kpt_fn2] = data2

            kpts2 = data2['keypoints']
            depth2 = data2['depth']

            if kpts2.shape[0] < 1024:
                continue

            full_ids1 = np.array([v for v in range(kpts1.shape[0])])
            valid_kpt_ids1 = (depth1 > 0)
            valid_kpts1 = kpts1[valid_kpt_ids1]
            valid_depth1 = depth1[valid_kpt_ids1]
            valid_ids1 = full_ids1[valid_kpt_ids1]

            full_ids2 = np.array([v for v in range(kpts2.shape[0])])
            valid_kpt_ids2 = (depth2 > 0)
            valid_kpts2 = kpts2[valid_kpt_ids2]
            valid_depth2 = depth2[valid_kpt_ids2]
            valid_ids2 = full_ids2[valid_kpt_ids2]

            if valid_ids1.shape[0] <= 20 or valid_ids2.shape[0] <= 20:
                continue

            # kpts1 = valid_kpts1
            # depth1 = valid_depth1

            # kpts2 = valid_kpts2
            # depth2 = valid_depth2

            # inlier_matches, outlier_matches = match_from_projection_points(
            #     pos1=kpts1.transpose(), depth1=depth1, intrinsics1=intrinsics1, pose1=pose1, bbox1=None,
            #     pos2=kpts2.transpose(), depth2=depth2, intrinsics2=intrinsics2, pose2=pose2, bbox2=None,
            #     inlier_th=self.inlier_th, outlier_th=15, cycle_check=True,
            # )

            with torch.no_grad():
                inlier_matches, outlier_matches = match_from_projection_points_torch(
                    pos1=torch.from_numpy(valid_kpts1.transpose()).float().cuda(),
                    depth1=torch.from_numpy(valid_depth1).float().cuda(),
                    intrinsics1=torch.from_numpy(intrinsics1).float().cuda(),
                    pose1=torch.from_numpy(pose1).float().cuda(),
                    bbox1=None,
                    pos2=torch.from_numpy(valid_kpts2.transpose()).float().cuda(),
                    depth2=torch.from_numpy(valid_depth2).float().cuda(),
                    intrinsics2=torch.from_numpy(intrinsics2).float().cuda(),
                    pose2=torch.from_numpy(pose2).float().cuda(),
                    bbox2=None,
                    inlier_th=5, outlier_th=15, cycle_check=True,
                )
                inlier_matches = inlier_matches.cpu().numpy()
                outlier_matches = outlier_matches.cpu().numpy()

            # print('valid1/2: ', valid_ids1.shape, valid_ids2.shape, inlier_matches.shape)

            if inlier_matches.shape[0] <= 20:
                continue

            matched_ids1 = []
            matched_ids2 = []
            for m in inlier_matches:
                if valid_ids1[m[0]] in matched_ids1 or valid_ids2[m[1]] in matched_ids2:
                    continue
                matched_ids1.append(valid_ids1[m[0]])
                matched_ids2.append(valid_ids2[m[1]])

            inlier_matches = np.array([matched_ids1, matched_ids2], dtype=int).transpose()
            # print(inlier_matches.shape)

            if show_match:
                img1 = cv2.imread(osp.join(self.base_path, image_path1))
                img2 = cv2.imread(osp.join(self.base_path, image_path2))
                img_match = plot_matches_cv2(image0=img1, image1=img2, kpts0=kpts1, kpts1=kpts2, matches=inlier_matches)
                cv2.imshow('img', img_match)
                cv2.waitKey(0)

            valid_pairs.append({
                'image_path1': image_paths[idx1],
                'depth_path1': depth_paths[idx1],
                'intrinsics1': intrinsics[idx1],
                'pose1': poses[idx1],
                'image_path2': image_paths[idx2],
                'depth_path2': depth_paths[idx2],
                'intrinsics2': intrinsics[idx2],
                'pose2': poses[idx2],
                'matched_ids1': np.array(matched_ids1, dtype=int),
                'matched_ids2': np.array(matched_ids2, dtype=int),
                # 'umatched_ids1': np.array(umatched_ids1, dtype=int),
                # 'umatched_ids2': np.array(umatched_ids2, dtype=int),
            })

            # if len(valid_pairs) > 10:
            #     break

        if len(valid_pairs) > 0:
            np.save(osp.join(match_dir, scene), valid_pairs)
        print('Find {:d}/{:d} valid pairs from scene {:s}'.format(len(valid_pairs), len(selected_ids), scene))
        scene_nvalid = {}
        scene_nvalid[scene] = len(valid_pairs)
        np.save('{:s}_nvalid_{:s}'.format(scene, self.feature_type), scene_nvalid)
        del all_keypoints

    def write_matches(self, save_dir, scene_list):
        # root = '/scratches/flyer_3/fx221/dataset/Megadepth/training_data'
        # match_dir = osp.join(root, 'matches_20220512_v1')
        # save_root = osp.join(root, 'matches_20220512_v1_sep')

        # root = '/scratches/flyer_2/fx221/dataset/Megadepth/training_data'
        match_dir = osp.join(save_dir, 'matches_{:s}'.format(self.feature_type))
        save_root = osp.join(save_dir, 'matches_sep_{:s}'.format(self.feature_type))

        for fn in tqdm(scene_list, total=len(scene_list)):
            if not osp.isfile(osp.join(match_dir, fn + ".npy")):
                continue
            print('Process: ', fn)
            data = np.load(osp.join(match_dir, fn + ".npy"), allow_pickle=True)

            save_dir_scene = osp.join(save_root, fn.split('.')[0])
            if not osp.exists(save_dir_scene):
                os.makedirs(save_dir_scene)
            for idx, d in tqdm(enumerate(data), total=len(data)):
                np.save(osp.join(save_dir_scene, '{:d}'.format(idx)), d)

    def extract_image_fns(self):
        for scene in tqdm(self.scenes, total=len(self.scenes)):
            scene_info_path = osp.join(self.scene_info_path, '{:s}.0.npz'.format(scene))
            if not osp.exists(scene_info_path):
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)

            image_paths = scene_info['image_paths']
            depth_paths = scene_info['depth_paths']
            intrinsics = scene_info['intrinsics']
            poses = scene_info['poses']

            assert len(image_paths) == len(depth_paths)
            assert len(image_paths) == len(intrinsics)
            assert len(image_paths) == len(poses)

            # print('Find {:d} images in scene {:s}'.format(len(image_paths), scene))
            for ni in range(len(image_paths)):
                image_path = image_paths[ni]
                depth_path = depth_paths[ni]
                pose = poses[ni]
                intrinsic = intrinsics[ni]

                if image_path is not None and depth_path is not None and pose is not None and intrinsic is not None:
                    self.image_paths.append(image_path)
                    self.depth_paths.append(depth_path)
                    self.poses.append(pose)
                    self.intrinsics.append(intrinsic)

        print('Find {:d} images in total'.format(len(self.image_paths)))


if __name__ == '__main__':
    feat_type = 'spp'  # 'sift'
    base_path = '/scratches/flyer_3/fx221/dataset/Megadepth'
    scene_list_fn = 'assets/megadepth_scenes_full.txt'

    scenes = []
    with open(scene_list_fn, 'r') as f:
        lines = f.readlines()
        for l in lines:
            scenes.append(l.strip())

    scene_info_path = osp.join(base_path, 'scene_info')
    mega = Megadepth(scene_info_path=scene_info_path,
                     base_path=base_path,
                     scene_list_fn=scene_list_fn,
                     nfeatures=4096,
                     # feature_type='sift',
                     feature_type=feat_type,
                     # feature_type=None,
                     # extract_features=False,
                     extract_features=True,
                     inlier_th=5,
                     min_overlap_ratio=0.1,
                     max_overlap_ratio=0.8,
                     )

    loader = torch.utils.data.DataLoader(dataset=mega,
                                         num_workers=8,
                                         shuffle=False,
                                         batch_size=1,
                                         pin_memory=True,
                                         )

    save_dir = '/scratches/flyer_3/fx221/dataset/Megadepth/training_data'
    save_dir_keypoint = osp.join(save_dir, 'keypoints_{:s}'.format(feat_type))

    print('Start extracting keypoints...')
    for bid, data in tqdm(enumerate(loader), total=len(loader)):
        image_path = data['image_path'][0]
        depth_path = data['depth_path'][0]
        keypoints = data['keypoints'][0].numpy()
        scores = data['scores'][0].numpy()
        descriptors = data['descriptors'][0].numpy()
        image_size = data['image_size'][0].numpy()
        depth = data['depth'][0].numpy()
        pose = data['pose'][0].numpy()
        intrinsics = data['intrinsics'][0].numpy()

        image_paths = image_path.split('/')
        scene = image_paths[1]
        img_fn = image_paths[-1]

        if not osp.exists(osp.join(save_dir_keypoint, scene)):
            os.makedirs(osp.join(save_dir_keypoint, scene))

        save_fn = osp.join(save_dir_keypoint, scene, img_fn + '_{:s}'.format(feat_type))
        save_data = {
            'image_path': image_path,
            'depth_path': depth_path,
            'intrinsics': intrinsics,
            'pose': pose,
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            'image_size': image_size,
            'depth': depth,
        }
        np.save(save_fn, save_data)

    print('Finish extracting keypoints...')

    print('Start building correspondences...')
    scene_npairs = []
    for s in scenes:
        s_pairs = mega.build_correspondence(scene=s, save_dir=save_dir)
        mega.write_matches(scene_list=[s])

    # merge scene-pairs to a single file
    mega_scene_pairs = {}
    for d in scene_npairs:
        mega_scene_pairs = {**mega_scene_pairs, **d}
    np.save('asserts/mega_nvalid_{:s}'.format(feat_type), mega_scene_pairs)
    
    print('Finish building correspondences...')
