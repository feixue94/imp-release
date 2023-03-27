import os
import glob
import numpy as np
import h5py
from .base_dumper import BaseDumper, np_skew_symmetric

import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)


# import utils


class scannet(BaseDumper):
    def get_seqs(self):
        self.pair_list = np.loadtxt('../assets/scannet_eval_list.txt', dtype=str)
        self.seq_list = np.unique(np.asarray([path.split('/')[0] for path in self.pair_list[:, 0]], dtype=str))
        self.dump_seq, self.img_seq = [], []
        for seq in self.seq_list:
            dump_dir = os.path.join(self.config['feature_dump_dir'], seq)
            cur_img_seq = glob.glob(os.path.join(os.path.join(self.config['rawdata_dir'], seq, 'img', '*.jpg')))
            cur_dump_seq = [
                os.path.join(dump_dir, path.split('/')[-1]) + '_' + self.config['extractor']['name'] + '_' + str(
                    self.config['extractor']['num_kpt']) \
                + '.hdf5' for path in cur_img_seq]
            self.img_seq += cur_img_seq
            self.dump_seq += cur_dump_seq

    def format_dump_folder(self):
        if not os.path.exists(self.config['feature_dump_dir']):
            os.mkdir(self.config['feature_dump_dir'])
        for seq in self.seq_list:
            seq_dir = os.path.join(self.config['feature_dump_dir'], seq)
            if not os.path.exists(seq_dir):
                os.mkdir(seq_dir)

    def format_dump_data(self):
        print('Formatting data...')
        self.data = {'K1': [], 'K2': [], 'R': [], 'T': [], 'e': [], 'f': [], 'fea_path1': [], 'fea_path2': [],
                     'img_path1': [], 'img_path2': []}

        for pair in self.pair_list:
            img_path1, img_path2 = pair[0], pair[1]
            seq = img_path1.split('/')[0]
            index1, index2 = int(img_path1.split('/')[-1][:-4]), int(img_path2.split('/')[-1][:-4])
            ex1, ex2 = np.loadtxt(os.path.join(self.config['rawdata_dir'], seq, 'extrinsic', str(index1) + '.txt'),
                                  dtype=float), \
                       np.loadtxt(os.path.join(self.config['rawdata_dir'], seq, 'extrinsic', str(index2) + '.txt'),
                                  dtype=float)
            K1, K2 = np.loadtxt(os.path.join(self.config['rawdata_dir'], seq, 'intrinsic', str(index1) + '.txt'),
                                dtype=float), \
                     np.loadtxt(os.path.join(self.config['rawdata_dir'], seq, 'intrinsic', str(index2) + '.txt'),
                                dtype=float)

            relative_extrinsic = np.matmul(np.linalg.inv(ex2), ex1)
            dR, dt = relative_extrinsic[:3, :3], relative_extrinsic[:3, 3]
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

            dump_seq_dir = os.path.join(self.config['feature_dump_dir'], seq)
            fea_path1, fea_path2 = os.path.join(dump_seq_dir,
                                                img_path1.split('/')[-1] + '_' + self.config['extractor']['name']
                                                + '_' + str(self.config['extractor']['num_kpt']) + '.hdf5'), \
                                   os.path.join(dump_seq_dir,
                                                img_path2.split('/')[-1] + '_' + self.config['extractor']['name']
                                                + '_' + str(self.config['extractor']['num_kpt']) + '.hdf5')
            self.data['img_path1'].append(img_path1), self.data['img_path2'].append(img_path2)
            self.data['fea_path1'].append(fea_path1), self.data['fea_path2'].append(fea_path2)

        self.form_standard_dataset()
