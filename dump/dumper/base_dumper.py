from abc import ABCMeta, abstractmethod
import os
import h5py
import numpy as np
from tqdm import trange
from torch.multiprocessing import Pool, set_start_method

set_start_method('spawn', force=True)

import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)
from components import load_component


def np_skew_symmetric(v):
    zero = np.zeros_like(v[:, 0])
    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)
    return M


class BaseDumper(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config
        self.img_seq = []
        self.dump_seq = []  # feature dump seq

    @abstractmethod
    def get_seqs(self):
        raise NotImplementedError

    @abstractmethod
    def format_dump_folder(self):
        raise NotImplementedError

    @abstractmethod
    def format_dump_data(self):
        raise NotImplementedError

    def initialize(self):
        self.extractor = load_component('extractor', self.config['extractor']['name'], self.config['extractor'])
        self.get_seqs()
        self.format_dump_folder()

    def extract(self, index):
        img_path, dump_path = self.img_seq[index], self.dump_seq[index]
        if not self.config['extractor']['overwrite'] and os.path.exists(dump_path):
            return
        kp, desc = self.extractor.run(img_path)
        self.write_feature(kp, desc, dump_path)

    def dump_feature(self):
        print('Extrating features...')
        self.num_img = len(self.dump_seq)
        pool = Pool(self.config['extractor']['num_process'])
        iteration_num = self.num_img // self.config['extractor']['num_process']
        if self.num_img % self.config['extractor']['num_process'] != 0:
            iteration_num += 1
        for index in trange(iteration_num):
            indicies_list = range(index * self.config['extractor']['num_process'],
                                  min((index + 1) * self.config['extractor']['num_process'], self.num_img))
            pool.map(self.extract, indicies_list)
        pool.close()
        pool.join()

    def write_feature(self, pts, desc, filename):
        with h5py.File(filename, "w") as ifp:
            ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
            ifp.create_dataset('descriptors', desc.shape, dtype=np.float32)
            ifp["keypoints"][:] = pts
            ifp["descriptors"][:] = desc

    def form_standard_dataset(self):
        dataset_path = os.path.join(self.config['dataset_dump_dir'], self.config['data_name'] + \
                                    '_' + self.config['extractor']['name'] + '_' + str(
            self.config['extractor']['num_kpt']) + '.hdf5')

        pair_data_type = ['K1', 'K2', 'R', 'T', 'e', 'f']
        num_pairs = len(self.data['K1'])
        with h5py.File(dataset_path, 'w') as f:
            print('collecting pair info...')
            for type in pair_data_type:
                dg = f.create_group(type)
                for idx in range(num_pairs):
                    data_item = np.asarray(self.data[type][idx])
                    dg.create_dataset(str(idx), data_item.shape, data_item.dtype, data=data_item)

            for type in ['img_path1', 'img_path2']:
                dg = f.create_group(type)
                for idx in range(num_pairs):
                    dg.create_dataset(str(idx), [1], h5py.string_dtype(encoding='ascii'),
                                      data=self.data[type][idx].encode('ascii'))

            # dump desc
            print('collecting desc and kpt...')
            desc1_g, desc2_g, kpt1_g, kpt2_g = f.create_group('desc1'), f.create_group('desc2'), f.create_group(
                'kpt1'), f.create_group('kpt2')
            for idx in trange(num_pairs):
                desc_file1, desc_file2 = h5py.File(self.data['fea_path1'][idx], 'r'), h5py.File(
                    self.data['fea_path2'][idx], 'r')
                desc1, desc2, kpt1, kpt2 = desc_file1['descriptors'][()], desc_file2['descriptors'][()], \
                                           desc_file1['keypoints'][()], desc_file2['keypoints'][()]
                desc1_g.create_dataset(str(idx), desc1.shape, desc1.dtype, data=desc1)
                desc2_g.create_dataset(str(idx), desc2.shape, desc2.dtype, data=desc2)
                kpt1_g.create_dataset(str(idx), kpt1.shape, kpt1.dtype, data=kpt1)
                kpt2_g.create_dataset(str(idx), kpt2.shape, kpt2.dtype, data=kpt2)
