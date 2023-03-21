# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> viz_utils
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   09/03/2022 14:42
=================================================='''
import cv2
import numpy as np
from copy import deepcopy


def plot_kpts(img, kpts, radius=None, colors=None, r=3, color=(0, 0, 255), nh=-1, nw=-1, shape='o', show_text=None, thickness=5):
    img_out = deepcopy(img)
    for i in range(kpts.shape[0]):
        pt = kpts[i]
        if radius is not None:
            if shape == 'o':
                img_out = cv2.circle(img_out, center=(int(pt[0]), int(pt[1])), radius=radius[i],
                                     color=color if colors is None else colors[i],
                                     thickness=thickness)
            elif shape == '+':
                img_out = cv2.line(img_out, pt1=(int(pt[0] - radius[i]), int(pt[1])),
                                   pt2=(int(pt[0] + radius[i]), int(pt[1])),
                                   color=color if colors is None else colors[i],
                                   thickness=5)
                img_out = cv2.line(img_out, pt1=(int(pt[0]), int(pt[1] - radius[i])),
                                   pt2=(int(pt[0]), int(pt[1] + radius[i])), color=color,
                                   thickness=thickness)
        else:
            if shape == 'o':
                img_out = cv2.circle(img_out, center=(int(pt[0]), int(pt[1])), radius=r,
                                     color=color if colors is None else colors[i],
                                     thickness=thickness)
            elif shape == '+':
                img_out = cv2.line(img_out, pt1=(int(pt[0] - r), int(pt[1])),
                                   pt2=(int(pt[0] + r), int(pt[1])), color=color if colors is None else colors[i],
                                   thickness=thickness)
                img_out = cv2.line(img_out, pt1=(int(pt[0]), int(pt[1] - r)),
                                   pt2=(int(pt[0]), int(pt[1] + r)), color=color if colors is None else colors[i],
                                   thickness=thickness)

    if show_text is not None:
        img_out = cv2.putText(img_out, show_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                              (0, 0, 255), 3)
    if nh == -1 and nw == -1:
        return img_out
    if nh > 0:
        return cv2.resize(img_out, dsize=(int(img.shape[1] / img.shape[0] * nh), nh))
    if nw > 0:
        return cv2.resize(img_out, dsize=(nw, int(img.shape[0] / img.shape[1] * nw)))
