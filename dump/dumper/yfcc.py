import os
import glob
import pickle
import numpy as np
import h5py
from .base_dumper import BaseDumper, np_skew_symmetric

import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)


class yfcc(BaseDumper):

    def get_seqs(self):
        data_dir = os.path.join(self.config['rawdata_dir'], 'yfcc100m')
        for seq in self.config['data_seq']:
            for split in self.config['data_split']:
                split_dir = os.path.join(data_dir, seq, split)
                dump_dir = os.path.join(self.config['feature_dump_dir'], seq, split)
                cur_img_seq = glob.glob(os.path.join(split_dir, 'images', '*.jpg'))
                cur_dump_seq = [
                    os.path.join(dump_dir, path.split('/')[-1]) + '_' + self.config['extractor']['name'] + '_' + str(
                        self.config['extractor']['num_kpt']) \
                    + '.hdf5' for path in cur_img_seq]
                self.img_seq += cur_img_seq
                self.dump_seq += cur_dump_seq

    def format_dump_folder(self):
        if not os.path.exists(self.config['feature_dump_dir']):
            os.mkdir(self.config['feature_dump_dir'])
        for seq in self.config['data_seq']:
            seq_dir = os.path.join(self.config['feature_dump_dir'], seq)
            if not os.path.exists(seq_dir):
                os.mkdir(seq_dir)
            for split in self.config['data_split']:
                split_dir = os.path.join(seq_dir, split)
                if not os.path.exists(split_dir):
                    os.mkdir(split_dir)

    def format_dump_data(self):
        print('Formatting data...')
        pair_path = os.path.join(self.config['rawdata_dir'], 'pairs')
        self.data = {'K1': [], 'K2': [], 'R': [], 'T': [], 'e': [], 'f': [], 'fea_path1': [], 'fea_path2': [],
                     'img_path1': [], 'img_path2': []}

        for seq in self.config['data_seq']:
            pair_name = os.path.join(pair_path, seq + '-te-1000-pairs.pkl')
            with open(pair_name, 'rb') as f:
                pairs = pickle.load(f)

            # generate id list
            seq_dir = os.path.join(self.config['rawdata_dir'], 'yfcc100m', seq, 'test')
            name_list = np.loadtxt(os.path.join(seq_dir, 'images.txt'), dtype=str)
            cam_name_list = np.loadtxt(os.path.join(seq_dir, 'calibration.txt'), dtype=str)

            for cur_pair in pairs:
                index1, index2 = cur_pair[0], cur_pair[1]
                cam1, cam2 = h5py.File(os.path.join(seq_dir, cam_name_list[index1]), 'r'), h5py.File(
                    os.path.join(seq_dir, cam_name_list[index2]), 'r')
                K1, K2 = cam1['K'][()], cam2['K'][()]
                [w1, h1], [w2, h2] = cam1['imsize'][()][0], cam2['imsize'][()][0]
                cx1, cy1, cx2, cy2 = (w1 - 1.0) * 0.5, (h1 - 1.0) * 0.5, (w2 - 1.0) * 0.5, (h2 - 1.0) * 0.5
                K1[0, 2], K1[1, 2], K2[0, 2], K2[1, 2] = cx1, cy1, cx2, cy2

                R1, R2, t1, t2 = cam1['R'][()], cam2['R'][()], cam1['T'][()].reshape([3, 1]), cam2['T'][()].reshape(
                    [3, 1])
                dR = np.dot(R2, R1.T)
                dt = t2 - np.dot(dR, t1)
                dt /= np.sqrt(np.sum(dt ** 2))

                e_gt_unnorm = np.reshape(np.matmul(
                    np.reshape(np_skew_symmetric(dt.astype('float64').reshape(1, 3)), (3, 3)),
                    np.reshape(dR.astype('float64'), (3, 3))), (3, 3))
                e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)
                f_gt_unnorm = np.linalg.inv(K2.T) @ e_gt @ np.linalg.inv(K1)
                f_gt = f_gt_unnorm / np.linalg.norm(f_gt_unnorm)

                self.data['K1'].append(K1), self.data['K2'].append(K2)
                self.data['R'].append(dR), self.data['T'].append(dt)
                self.data['e'].append(e_gt), self.data['f'].append(f_gt)

                img_path1, img_path2 = os.path.join('yfcc100m', seq, 'test', name_list[index1]), os.path.join(
                    'yfcc100m', seq, 'test', name_list[index2])
                dump_seq_dir = os.path.join(self.config['feature_dump_dir'], seq, 'test')
                fea_path1, fea_path2 = os.path.join(dump_seq_dir,
                                                    name_list[index1].split('/')[-1] + '_' + self.config['extractor'][
                                                        'name']
                                                    + '_' + str(self.config['extractor']['num_kpt']) + '.hdf5'), \
                                       os.path.join(dump_seq_dir,
                                                    name_list[index2].split('/')[-1] + '_' + self.config['extractor'][
                                                        'name']
                                                    + '_' + str(self.config['extractor']['num_kpt']) + '.hdf5')
                self.data['img_path1'].append(img_path1), self.data['img_path2'].append(img_path2)
                self.data['fea_path1'].append(fea_path1), self.data['fea_path2'].append(fea_path2)

        self.form_standard_dataset()
