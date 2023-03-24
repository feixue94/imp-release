import os
import numpy as np
import h5py
import cv2
from torch.utils.data import Dataset


class standard_reader:
    def __init__(self, config):
        self.raw_dir = config['rawdata_dir']
        self.dataset = h5py.File(config['dataset_dir'], 'r')
        self.num_kpt = config['num_kpt']

    def run(self, index):
        K1, K2 = np.asarray(self.dataset['K1'][str(index)]), np.asarray(self.dataset['K2'][str(index)])
        R = np.asarray(self.dataset['R'][str(index)])
        t = np.asarray(self.dataset['T'][str(index)])
        t = t / np.sqrt((t ** 2).sum())

        desc1, desc2 = self.dataset['desc1'][str(index)][()][:self.num_kpt], self.dataset['desc2'][str(index)][()][
                                                                             :self.num_kpt]
        x1, x2 = self.dataset['kpt1'][str(index)][()][:self.num_kpt], self.dataset['kpt2'][str(index)][()][
                                                                      :self.num_kpt]
        e, f = self.dataset['e'][str(index)][()], self.dataset['f'][str(index)][()]

        img1_path, img2_path = self.dataset['img_path1'][str(index)][()][0].decode(), \
                               self.dataset['img_path2'][str(index)][()][0].decode()
        img1, img2 = cv2.imread(os.path.join(self.raw_dir, img1_path)), cv2.imread(
            os.path.join(self.raw_dir, img2_path))

        info = {'index': index, 'K1': K1, 'K2': K2, 'R': R, 't': t, 'x1': x1, 'x2': x2, 'desc1': desc1, 'desc2': desc2,
                'img1': img1, 'img2': img2, 'e': e, 'f': f, 'r_gt': R, 't_gt': t}
        return info

    def close(self):
        self.dataset.close()

    def __len__(self):
        return len(self.dataset['K1'])


class reader_set(Dataset):
    def __init__(self, config):
        self.raw_dir = config['rawdata_dir']
        self.dataset = h5py.File(config['dataset_dir'], 'r')
        self.num_kpt = config['num_kpt']

    def __getitem__(self, index):
        K1, K2 = np.asarray(self.dataset['K1'][str(index)]), np.asarray(self.dataset['K2'][str(index)])
        R = np.asarray(self.dataset['R'][str(index)])
        t = np.asarray(self.dataset['T'][str(index)])
        t = t / np.sqrt((t ** 2).sum())

        desc1, desc2 = self.dataset['desc1'][str(index)][()][:self.num_kpt], self.dataset['desc2'][str(index)][()][
                                                                             :self.num_kpt]
        x1, x2 = self.dataset['kpt1'][str(index)][()][:self.num_kpt], self.dataset['kpt2'][str(index)][()][
                                                                      :self.num_kpt]
        e, f = self.dataset['e'][str(index)][()], self.dataset['f'][str(index)][()]

        img1_path, img2_path = self.dataset['img_path1'][str(index)][()][0].decode(), \
                               self.dataset['img_path2'][str(index)][()][0].decode()
        img1, img2 = cv2.imread(os.path.join(self.raw_dir, img1_path)), cv2.imread(
            os.path.join(self.raw_dir, img2_path))

        info = {'index': index, 'K1': K1, 'K2': K2, 'R': R, 't': t, 'x1': x1, 'x2': x2, 'desc1': desc1, 'desc2': desc2,
                'img1': img1, 'img2': img2, 'e': e, 'f': f, 'r_gt': R, 't_gt': t}
        return info

    def close(self):
        self.dataset.close()

    def __len__(self):
        return len(self.dataset['K1'])
