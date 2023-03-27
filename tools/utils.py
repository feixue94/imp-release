# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import copy
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy

# import pymagsac
matplotlib.use('Agg')


# matplotlib.use('TkAgg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1. / total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """

    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, False)
        if self.camera:

            if self._ip_camera:
                # Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file)
        self.i = self.i + 1
        return (image, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            # print('IPCAMERA THREAD got frame {}'.format(self._ip_index))

    def cleanup(self):
        self._ip_running = False


# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert (len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame):
    return torch.from_numpy(frame / 255.).float()[None, None].cuda()


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    # inp = frame2tensor(image, device)
    inp = frame2tensor(image)
    return image, inp, scales


def read_image_modified(image, resize, resize_float):
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w - 1 - cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w - 1 - cx],
                         [0., fy, h - 1 - cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h - 1 - cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1. / scales[0], 1. / scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
                      + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))
    return d


def compute_epipolar_error_pixel(kpts0, kpts1, T_0to1, K0, K1):
    # kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    # kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    dist = np.sqrt(t0 ** 2 + t1 ** 2 + t2 ** 2)
    t0 = t0 / dist
    t1 = t1 / dist
    t2 = t2 / dist

    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)

    Ep0 = kpts0 @ F.T  # N x 3
    p1Ep0 = np.sum((kpts1 * Ep0) ** 2, -1)  # N
    Etp1 = kpts1 @ F  # N x 3
    # d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
    #                   + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))
    return p1Ep0


def compute_epipolar_sampson_dist(kpts0, kpts1, T_0to1, K0, K1):
    # print(kpts0.shape, kpts1.shape, T_0to1.shape, K0.shape, K1.shape)

    kpts1_homo = np.concatenate([kpts0, np.ones_like(kpts0[:, :1])], axis=-1)
    kpts2_homo = np.concatenate([kpts1, np.ones_like(kpts1[:, :1])], axis=-1)

    t0, t1, t2 = T_0to1[:3, 3]
    # print(t0, t1, t2)
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)

    Fx1 = F @ kpts1_homo.T  # [3, N]
    Fx2 = (kpts2_homo @ F).T  # [3, N]
    demo = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    err = np.diag((kpts2_homo @ (F @ kpts1_homo.T))) ** 2 / demo  # sampson error
    return err


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


# --- VISUALIZATION ---


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size * n, size * 3 / 4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2, ms=4):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
        for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches_cv2(image0, image1, kpts0, kpts1, pred_matches, gt_matches=None, save_fn=None, margin=10,
                     vis_baseline=None, plot_keypoints=True, inliers=None):
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

    if plot_keypoints:
        for p0 in kpts0:
            match_img = cv2.circle(match_img, center=(int(p0[0]), int(p0[1])), radius=3, color=(0, 0, 255), thickness=2)
        for p1 in kpts1:
            match_img = cv2.circle(match_img, center=(int(p1[0] + w0 + margin), int(p1[1])), radius=3,
                                   color=(0, 0, 255),
                                   thickness=2)

    if gt_matches is not None:
        match_img_gt = copy.copy(match_img)
        n_gt_matches = 0
        for id0 in range(gt_matches.shape[0]):
            if gt_matches[id0] < 0:
                continue
            color = (0, 255, 0)
            p0 = kpts0[id0]
            p1 = kpts1[gt_matches[id0]]
            match_img_gt = cv2.circle(match_img_gt, center=(int(p0[0]), int(p0[1])), radius=3, color=(0, 0, 255),
                                      thickness=2)
            match_img_gt = cv2.circle(match_img_gt, center=(int(p1[0] + w0 + margin), int(p1[1])), radius=3,
                                      color=(0, 0, 255),
                                      thickness=2)
            match_img_gt = cv2.line(match_img_gt, pt1=(int(p0[0]), int(p0[1])),
                                    pt2=(int(p1[0] + w0 + margin), int(p1[1])),
                                    color=color, thickness=2)
            n_gt_matches += 1
        match_img_gt = cv2.putText(match_img_gt, '{:d}/{:d}'.format(n_gt_matches, gt_matches.shape[0]), (20, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 255), 2)

        # print('match_img_gt: ', match_img_gt.shape)
    else:
        match_img_gt = None

    n_corr = 0
    # for id0, id1 in enumerate(pred_matches):
    # for id0 in range(gt_matches.shape[0]):
    for id0 in range(pred_matches.shape[0]):
        if inliers is not None:
            if not inliers[id0]:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        # id1 = gt_matches[id0]
        id1 = pred_matches[id0]

        if id1 == -1:
            continue

        if gt_matches is not None:
            if id1 != gt_matches[id0]:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
                n_corr += 1

        # print(id0, id1)
        p0 = kpts0[id0]
        p1 = kpts1[id1]

        match_img = cv2.circle(match_img, center=(int(p0[0]), int(p0[1])), radius=3, color=(0, 0, 255), thickness=2)
        match_img = cv2.circle(match_img, center=(int(p1[0] + w0 + margin), int(p1[1])), radius=3,
                               color=(0, 0, 255),
                               thickness=2)
        match_img = cv2.line(match_img, pt1=(int(p0[0]), int(p0[1])), pt2=(int(p1[0] + w0 + margin), int(p1[1])),
                             color=color, thickness=2)

    n_pred_total = np.sum(pred_matches != -1)
    # in case
    if n_pred_total == 0:
        n_pred_total = 1
    if gt_matches is not None:
        n_gt_total = np.sum(gt_matches != -1)
        if n_gt_total == 0:
            n_gt_total = 1
        text = "{:d}/{:d}/{:.3f}/{:.3f}".format(n_pred_total, n_corr, n_corr / n_pred_total, n_corr / n_gt_total)
    else:
        n_gt_total = 0
        text = "{:d}/{:d}/{:.3f}".format(n_pred_total, n_corr, n_corr / n_pred_total)

    match_img = cv2.putText(match_img, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
    if match_img_gt is not None:
        match_img = np.vstack([match_img, match_img_gt])

    match_img = cv2.resize(match_img, dsize=None, fx=0.5, fy=0.5)

    if save_fn is not None:
        cv2.imwrite(save_fn, match_img)
    out = {
        'match_img': match_img,
        'inlier_ratio': n_corr / n_pred_total,
        'recall_ratio': n_corr / n_gt_total if gt_matches is not None else 0.,
        'n_corr_match': n_corr,
        'n_gt_match': n_gt_total,
    }
    return out


def plot_matches_spg(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text,
                     fast_viz=False, show_kpts=False, show_matches=True,
                     opencv_display=False, save_path=None
                     ):
    # color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
    # color = error_colormap(1 - color)
    # deg, delta = ' deg', 'Delta '
    # e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
    # e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
    # text = [
    #     'SuperGlue',
    #     '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
    #     'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
    # ]
    # print(image0.shape, kpts0.shape, kpts1.shape, mkpts0.shape, mkpts1.shape, color.shape)
    make_matching_plot(
        image0=image0, image1=image1, kpts0=kpts0, kpts1=kpts1, mkpts0=mkpts0,
        mkpts1=mkpts1, color=color, text=text, path=save_path,
        show_keypoints=show_kpts,
        show_matches=show_matches,
        fast_viz=fast_viz,
        opencv_display=opencv_display, small_text=['Relative Pose'])


# def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
#                        color, text, path, name0, name1, show_keypoints=False,
#                        fast_viz=False, opencv_display=False, opencv_title='matches'):
#
#     # print('fast_viz: ', fast_viz,)
#     # print('opencv_title: ', opencv_title)
#     if fast_viz:
#         make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
#                                 color, text, path, show_keypoints, 10,
#                                 opencv_display, opencv_title)
#         return
#
#     plot_image_pair([image0, image1])
#     if show_keypoints:
#         plot_keypoints(kpts0, kpts1, color='r', ps=4)
#         plot_keypoints(kpts0, kpts1, color='r', ps=2)
#     plot_matches(mkpts0, mkpts1, color)
#
#     fig = plt.gcf()
#     txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
#     fig.text(
#         0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
#         fontsize=15, va='top', ha='left', color=txt_color)
#
#     txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
#     fig.text(
#         0.01, 0.01, name0, transform=fig.axes[0].transAxes,
#         fontsize=5, va='bottom', ha='left', color=txt_color)
#
#     txt_color = 'k' if image1[-100:, :150].mean() > 200 else 'w'
#     fig.text(
#         0.01, 0.01, name1, transform=fig.axes[1].transAxes,
#         fontsize=5, va='bottom', ha='left', color=txt_color)
#
#     plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
#     plt.close()
#
#
# def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
#                             mkpts1, color, text, path=None,
#                             show_keypoints=False, margin=10,
#                             opencv_display=False, opencv_title=''):
#     H0, W0 = image0.shape
#     H1, W1 = image1.shape
#     H, W = max(H0, H1), W0 + W1 + margin
#
#     out = 255 * np.ones((H, W), np.uint8)
#     out[:H0, :W0] = image0
#     out[:H1, W0 + margin:] = image1
#     out = np.stack([out] * 3, -1)
#
#     if show_keypoints:
#         kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
#         white = (255, 255, 255)
#         black = (0, 0, 0)
#         for x, y in kpts0:
#             cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
#             cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
#         for x, y in kpts1:
#             cv2.circle(out, (x + margin + W0, y), 2, black, -1,
#                        lineType=cv2.LINE_AA)
#             cv2.circle(out, (x + margin + W0, y), 1, white, -1,
#                        lineType=cv2.LINE_AA)
#
#     mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
#     color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
#     for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
#         c = c.tolist()
#         cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
#                  color=c, thickness=1, lineType=cv2.LINE_AA)
#         # display line end-points as circles
#         cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
#         cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
#                    lineType=cv2.LINE_AA)
#
#     Ht = int(H * 30 / 480)  # text height
#     txt_color_fg = (255, 255, 255)
#     txt_color_bg = (0, 0, 0)
#     for i, t in enumerate(text):
#         cv2.putText(out, t, (10, Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
#                     H * 1.0 / 480, txt_color_bg, 2, cv2.LINE_AA)
#         cv2.putText(out, t, (10, Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
#                     H * 1.0 / 480, txt_color_fg, 1, cv2.LINE_AA)
#
#     if path is not None:
#         out = cv2.resize(out, dsize=None, fx=0.25, fy=0.25)
#         cv2.imwrite(str(path), out)
#
#     if opencv_display:
#         # print('opencv_display')
#         cv2.imshow(opencv_title, out)
#         cv2.waitKey(1)
#
#     return out


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False, show_matches=True,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):
    # print('fast_viz: ', fast_viz,)
    # print('opencv_title: ', opencv_title)
    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=20)  # 80
        plot_keypoints(kpts0, kpts1, color='r', ps=10)  # 60

    if show_matches:
        plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    if path is not None:
        # print(path)
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    plt.clf()


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin:] = image1
    out = np.stack([out] * 3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        nh = 512
        nw = int(out.shape[1] / out.shape[0] * nh)
        out_rs = cv2.resize(out, dsize=(nw, nh))
        cv2.imwrite(str(path), out_rs)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def error_colormap(x):
    return np.clip(
        np.stack([2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)


def eval_matches(pred_matches, gt_matches):
    """
    :param pred_matches:
    :param gt_matches:
    :return: inlier ration of prediced matches & recal ratio of inlier ratio
    """
    # pred_matches: [N, 2]
    # gt_matches: [M], -1: no matches

    n_corr = 0
    for i in range(pred_matches.shape[0]):
        id1 = pred_matches[i, 0]
        id2 = pred_matches[i, 1]
        if gt_matches[id1] == -1 or gt_matches[id1] != id2:
            continue

        n_corr += 1

    n_pred_total = pred_matches.shape[0]
    n_gt_total = np.sum(gt_matches != -1)
    return {
        'inlier_ratio': n_corr / n_pred_total if n_pred_total > 0 else 0,
        'recall_ratio': n_corr / n_gt_total if n_gt_total > 0 else 0,
    }
