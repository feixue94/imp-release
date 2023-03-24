# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> utils
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   03/04/2022 20:47
=================================================='''
import numpy as np


def normalize_size(x, size, scale=1):
    size = size.reshape([1, 2])
    norm_fac = size.max()
    return (x - size / 2 - 0.5) / (norm_fac * scale)


def normalize_size_spg(x, size, scale=0.7):
    size = size.reshape([1, 2])
    norm_fac = size.max()
    return (x - size / 2 - 0.5) / (norm_fac * scale)


def normalize_points_3d(x, scale=1.0):
    max_x = np.max(abs(x), axis=0)
    return x / (max_x * scale) - 0.5


def denormalize_points_3d(x, max_x, scale=1.0):
    return (x + 0.5) * (max_x * scale)
