# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> common
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/12/2021 15:54
=================================================='''

import os, pdb  # , shutil
import numpy as np
import torch
import json
from collections import OrderedDict
import cv2
from torch._six import string_classes
import collections.abc as collections
import os
import os.path as osp


def get_recursive_file_list(root_dir, sub_dir=None, patterns=[]):
    if sub_dir is not None:
        current_files = os.listdir(osp.join(root_dir, sub_dir))
    all_files = []
    for file_name in current_files:
        full_file_name = os.path.join(root_dir, sub_dir, file_name)
        print(file_name)

        if file_name.split('.')[-1] in patterns:
            all_files.append(osp.join(sub_dir, file_name))

        if os.path.isdir(full_file_name):
            next_level_files = get_recursive_file_list(root_dir, sub_dir=osp.join(sub_dir, file_name))
            all_files.extend(next_level_files)

    return all_files


def sort_dict_by_value(data, reverse=False):
    return sorted(data.items(), key=lambda d: d[1], reverse=reverse)


def mkdir_for(file_path):
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)


def model_size(model):
    ''' Computes the number of parameters of the model
    '''
    size = 0
    for weights in model.state_dict().values():
        size += np.prod(weights.shape)
    return size


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        # print(os.environ['CUDA_VISIBLE_DEVICES'])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True  # speed-up cudnn
        torch.backends.cudnn.fastest = True  # even more speed-up?
        print('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])

    else:
        print('Launching on CPU')

    return cuda


def save_args(args, save_path):
    with open(save_path, 'a+') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args(args, save_path):
    with open(save_path, "r") as f:
        args.__dict__ = json.load(f)


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def resize_img(img, nh=-1, nw=-1, mode=cv2.INTER_NEAREST):
    assert nh > 0 or nw > 0
    if nh == -1:
        return cv2.resize(img, dsize=(nw, int(img.shape[0] / img.shape[1] * nw)), interpolation=mode)
    if nw == -1:
        return cv2.resize(img, dsize=(int(img.shape[1] / img.shape[0] * nh), nh), interpolation=mode)
    return cv2.resize(img, dsize=(nw, nh), interpolation=mode)


def map_tensor(input_, func):
    if isinstance(input_, torch.Tensor):
        return func(input_)
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    else:
        raise TypeError(
            f'input must be tensor, dict or list; found {type(input_)}')
