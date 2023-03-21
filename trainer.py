# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> trainer
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/12/2021 15:05
=================================================='''

import datetime
import json
import os
import os.path as osp
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2
import torch.optim as optim
import shutil
import torch
from torch.autograd import Variable
from tools.common import save_args
from tools.utils import plot_matches_cv2, eval_matches

from nets.matcher import mutual_nn_matcher
from eval.eval_yfcc_full import evaluate_full


class Trainer:
    def __init__(self, model, train_loader, eval_loader=None, args=None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.args = args

        self.init_lr = self.args.lr
        self.min_lr = self.args.min_lr
        if self.args.optim == 'adam':
            self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                                        lr=self.init_lr)
        elif self.args.optim == 'adamw':
            self.optimizer = optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                         lr=self.init_lr)
        self.num_epochs = self.args.epochs

        if args.resume_path != 'None':
            log_dir = args.resume_path.split('/')[-2]
            resume_log = torch.load(osp.join(osp.join(args.save_root, args.resume_path)), map_location='cpu')
            self.epoch = resume_log['epoch'] + 1
            if 'iteration' in resume_log.keys():
                self.iteration = resume_log['iteration']
            else:
                self.iteration = len(self.train_loader) * self.epoch
            self.min_loss = resume_log['min_loss']
        else:
            self.iteration = 0
            self.epoch = 0
            self.min_loss = 1e10

            now = datetime.datetime.now()
            log_dir = now.strftime("%Y_%m_%d_%H_%M_%S")
            log_dir = log_dir + '_' + self.args.network + '_L' + str(
                self.args.layers) + '_' + self.args.dataset + '_' + str(self.args.feature) + '_B' + str(
                self.args.batch_size) + '_K' + str(self.args.max_keypoints) + '_M' + str(
                self.args.match_th) + '_' + self.args.ac_fn + '_' + self.args.norm_fn + '_' + self.args.optim

            if self.args.with_pose > 0:
                log_dir = log_dir + "_P{:d}".format(self.args.n_hypothesis)

        self.save_dir = osp.join(self.args.save_root, log_dir)
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        print("save_dir: ", self.save_dir)

        self.log_file = open(osp.join(self.save_dir, "log.txt"), "a+")
        if self.args.local_rank == 0:
            save_args(args=args, save_path=Path(self.save_dir, "args.txt"))
        self.writer = SummaryWriter(self.save_dir)

        self.tag = log_dir

        self.do_eval = (self.args.do_eval > 0)
        self.do_vis = self.args.vis

        if self.do_eval:
            self.eval_fun = self.eval_matching

        if self.do_vis and self.args.local_rank == 0:
            self.vis_dir = osp.join(self.save_dir, 'vis')
            if not osp.exists(self.vis_dir):
                os.makedirs(self.vis_dir)
            if self.do_eval:
                if not osp.exists(osp.join(self.vis_dir, 'eval')):
                    os.makedirs(osp.join(self.vis_dir, 'eval'))

    def process_epoch(self):
        self.model.train()

        epoch_losses = []
        epoch_acc_corr = []
        epoch_acc_incorr = []
        epoch_acc_corr_ratio = []
        epoch_acc_incorr_ratio = []
        epoch_matching_loss = []
        epoch_pose_loss = []
        epoch_valid_pose = []

        n_invalid_its = 0
        for bidx, pred in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            for k in pred:
                if k.find('file_name') >= 0:
                    continue
                if k != 'image0' and k != 'image1' and k != 'depth0' and k != 'depth1':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].float().cuda())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).float().cuda())
                # if k in ('image0_shape', 'image1_shape'):
                #     pred[k] = Variable(torch.stack(pred[k]).float().cuda())

            if self.args.its_per_epoch >= 0 and bidx >= self.args.its_per_epoch:
                break

            # my_context = self.model.no_sync if ar
            data = self.model(pred)
            for k, v in pred.items():
                pred[k] = v
            pred = {**pred, **data}

            loss = pred['loss']
            # loss_corr = pred['loss_corr']
            # loss_incorr = pred['loss_incorr']
            acc_corr = pred['acc_corr'][-1]
            acc_incorr = pred['acc_incorr'][-1]
            total_acc_corr = pred['total_acc_corr'][-1]
            total_acc_incorr = pred['total_acc_incorr'][-1]
            # print(loss, acc_corr, acc_incorr)
            # print('loss: ', loss, torch.numel(loss))

            if 'matching_loss' in pred.keys():
                matching_loss = pred['matching_loss']
            else:
                matching_loss = loss

            if 'pose_loss' in pred.keys():
                pose_loss = pred['pose_loss']
            else:
                pose_loss = loss

            if 'valid_pose' in pred.keys():
                valid_pose = pred['valid_pose']
            else:
                valid_pose = loss

            if torch.numel(loss) > 1:
                loss = torch.mean(loss)

                if torch.isinf(loss) or torch.isnan(loss):
                    # self.optimizer.zero_grad()
                    print('Loss is INF/NAN')
                    self.optimizer.zero_grad()
                    del pred
                    torch.cuda.empty_cache()
                    # del data
                    n_invalid_its += 1
                    if n_invalid_its >= 10:
                        print('Exit because of INF/NAN in loss')
                        # exit(0)
                        torch.cuda.empty_cache()
                        return None
                    continue

                matching_loss = torch.mean(matching_loss)
                pose_loss = torch.mean(pose_loss)
                valid_pose = torch.mean(valid_pose)

                # loss_corr = torch.mean(loss_corr)
                # loss_incorr = torch.mean(loss_incorr)
                acc_corr = torch.mean(acc_corr)
                acc_incorr = torch.mean(acc_incorr)
                total_acc_corr = torch.mean(total_acc_corr)
                total_acc_incorr = torch.mean(total_acc_incorr)
            else:
                if torch.isinf(loss) or torch.isnan(loss):
                    print('Loss is INF/NAN')
                    self.optimizer.zero_grad()
                    del pred
                    torch.cuda.empty_cache()
                    # del data
                    n_invalid_its += 1
                    if n_invalid_its >= 10:
                        print('Exit because of INF/NAN in loss')
                        # exit(0)
                        torch.cuda.empty_cache()
                        return None
                    continue

            epoch_losses.append(loss.item())
            epoch_matching_loss.append(matching_loss.item())
            epoch_pose_loss.append(pose_loss.item())
            epoch_valid_pose.append(valid_pose.item())

            epoch_acc_corr.append(acc_corr.item())
            epoch_acc_incorr.append(acc_incorr.item())
            acc_corr_ratio = acc_corr.item() / (total_acc_corr.item() + 1)
            acc_incorr_ratio = acc_incorr.item() / (total_acc_incorr.item() + 1)
            epoch_acc_corr_ratio.append(acc_corr_ratio)
            epoch_acc_incorr_ratio.append(acc_incorr_ratio)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iteration += 1

            lr = min(self.args.lr * self.args.decay_rate ** (self.iteration - self.args.decay_iter), self.args.lr)
            if lr < self.min_lr:
                lr = self.min_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if self.args.local_rank == 0 and bidx % self.args.log_intervals == 0:
                matching_score = pred['matching_scores0'][-1]
                # mean_matching_score = torch.mean(matching_score)
                # medien_matching_score = torch.median(matching_score)
                print_text = 'Epoch [{:d}/{:d}], Step [{:d}/{:d}/{:d}], Loss [m{:.2f}/p{:.2f}/t{:.2f}], MS [{:.2f}], Acc [c{:.1f}/{:.1f}, n{:.1f}/{:.1f}], P[{:.1f}]'.format(
                    self.epoch,
                    self.num_epochs, bidx,
                    len(self.train_loader),
                    self.iteration,
                    matching_loss.item(),
                    pose_loss.item(),
                    loss.item(),
                    torch.max(matching_score).item(),
                    np.mean(epoch_acc_corr),
                    np.mean(epoch_acc_corr_ratio),
                    np.mean(epoch_acc_incorr),
                    np.mean(epoch_acc_incorr_ratio),
                    np.mean(epoch_valid_pose))
                print(print_text)
                self.log_file.write(print_text + '\n')

                info = {
                    'lr': lr,
                    'matching_loss': matching_loss.item(),
                    'pose_loss': pose_loss.item(),
                    'loss': loss.item(),
                    'acc_corr': acc_corr.item(),
                    'acc_incorr': acc_incorr.item(),
                    'acc_corr_ratio': acc_corr_ratio,
                    'acc_incorr_ratio': acc_incorr_ratio,
                    'valid_pose': valid_pose.item(),
                }
                for k, v in info.items():
                    self.writer.add_scalar(tag=k, scalar_value=v, global_step=self.iteration)

                if self.do_vis:
                    image0, image1 = pred['image0'].cpu().numpy()[0], pred['image1'].cpu().numpy()[0]
                    if image0.shape[0] == 1:
                        image0 = image0[0]
                    if image1.shape[0] == 1:
                        image1 = image1[0]

                    image0 = image0.astype('float32')
                    image1 = image1.astype('float32')
                    if len(image0.shape) == 2:
                        image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2BGR)
                        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)

                    kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
                    matches = pred['matches0'].cpu().detach().numpy()[0]
                    conf = pred['matching_scores0'].cpu().detach().numpy()[0]
                    matching_mask = pred['matching_mask'][0].cpu().detach().numpy().astype(int)  # [N, M]
                    gt_matches = np.zeros_like(matches) - 1
                    for p0 in range(kpts0.shape[0]):
                        p1 = np.where(matching_mask[p0] == 1)[0]
                        if p1 >= kpts1.shape[0]:
                            continue
                        gt_matches[p0] = p1
                    eval_out = plot_matches_cv2(image0=image0, image1=image1,
                                                kpts0=kpts0, kpts1=kpts1,
                                                pred_matches=matches,
                                                gt_matches=gt_matches,
                                                save_fn=osp.join(self.vis_dir,
                                                                 '{:d}_{:d}_cv.png'.format(self.epoch, bidx))
                                                )

                    desc0 = pred['descriptors0'][0]
                    desc1 = pred['descriptors1'][0]
                    nnm_matches = mutual_nn_matcher(descriptors1=desc0, descriptors2=desc1)
                    nnm_out = eval_matches(pred_matches=nnm_matches, gt_matches=gt_matches)

                    info = {
                        'train_inlier_ratio': eval_out['inlier_ratio'],
                        'train_recall_ratio': eval_out['recall_ratio'],
                        'nnm_inlier_ratio': nnm_out['inlier_ratio'],
                        'nnm_recall_ratio': nnm_out['recall_ratio'],
                        'train-nnm_inlier_ratio': eval_out['inlier_ratio'] - nnm_out['inlier_ratio'],
                        'train-nnm_recall_ratio': eval_out['recall_ratio'] - nnm_out['recall_ratio'],
                        'mean_score': np.mean(conf),
                        'md_score': np.median(conf),
                    }
                    for k, v in info.items():
                        self.writer.add_scalar(tag=k, scalar_value=v, global_step=self.iteration)
                    self.writer.add_histogram(tag='score', values=matching_score, global_step=self.iteration)
        if self.args.local_rank == 0:
            print_text = 'Epoch [{:d}/{:d}], AVG Loss [m{:.2f}/p{:.2f}/t{:.2f}], Acc [c{:.1f}/{:.1f}, n{:.1f}/{:.1f}], P [{:.1f}]\n'.format(
                self.epoch,
                self.num_epochs,
                np.mean(epoch_matching_loss),
                np.mean(epoch_pose_loss),
                np.mean(epoch_losses),
                np.mean(epoch_acc_corr),
                np.mean(epoch_acc_corr_ratio),
                np.mean(epoch_acc_incorr),
                np.mean(epoch_acc_incorr_ratio),
                np.mean(epoch_valid_pose),
            )
            print(print_text)
            self.log_file.write(print_text + '\n')
            self.log_file.flush()
        return np.mean(epoch_losses)

    def eval(self):
        self.model.eval()
        print('Start to eval the model from epoch: {:d}'.format(self.epoch))

        mean_inlier_ratio = []
        mean_corr_matches = []
        mean_gt_matches = []
        mean_recall_ratio = []
        mean_loss = []

        for bidx, pred in tqdm(enumerate(self.eval_loader), total=len(self.eval_loader)):
            with torch.no_grad():
                for k in pred:
                    if k != 'file_name' and k != 'image0' and k != 'image1' and k != 'depth0' and k != 'depth1':
                        if type(pred[k]) == torch.Tensor:
                            pred[k] = Variable(pred[k].float().cuda())
                        else:
                            pred[k] = Variable(torch.stack(pred[k]).float().cuda())

                data = self.model(pred)
                for k, v in pred.items():
                    pred[k] = v
                pred = {**pred, **data}
                loss = pred['loss']
                if torch.numel(loss) > 1:
                    loss = torch.mean(loss)
                # print('loss: ', loss, type(loss))
                if torch.isinf(loss) or torch.isnan(loss):
                    # self.optimizer.zero_grad()
                    print('Loss is INF/NAN')
                    del pred
                    continue

                mean_loss.append(loss.item())
                image0, image1 = pred['image0'].cpu().numpy()[0], pred['image1'].cpu().numpy()[0]
                if image0.shape[0] == 1:
                    image0 = image0[0]
                if image1.shape[0] == 1:
                    image1 = image1[0]

                image0 = image0.astype('float32')
                image1 = image1.astype('float32')
                if len(image0.shape) == 2:
                    image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2BGR)
                    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)

                kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
                matches = pred['matches0'].cpu().detach().numpy()[0]
                conf = pred['matching_scores0'].cpu().detach().numpy()[0]
                matching_mask = pred['matching_mask'][0].cpu().detach().numpy().astype(int)  # [N, M]
                gt_matches = np.zeros_like(matches) - 1
                for p0 in range(kpts0.shape[0]):
                    p1 = np.where(matching_mask[p0] == 1)[0]
                    if p1 >= kpts1.shape[0]:
                        continue
                    gt_matches[p0] = p1
                eval_out = plot_matches_cv2(image0=image0, image1=image1,
                                            kpts0=kpts0, kpts1=kpts1,
                                            pred_matches=matches,
                                            gt_matches=gt_matches,
                                            save_fn=osp.join(osp.join(self.vis_dir, 'eval'),
                                                             '{:d}_{:d}_cv.png'.format(self.epoch,
                                                                                       bidx)) if bidx % self.args.log_intervals == 0 else None
                                            )

                mean_corr_matches.append(eval_out['n_corr_match'])
                mean_gt_matches.append(eval_out['n_gt_match'])
                mean_inlier_ratio.append(eval_out['inlier_ratio'])
                mean_recall_ratio.append(eval_out['recall_ratio'])

        print_text = 'Eval Epoch: {:d}, Loss: {:.2f}, Inlier_ratio:{:.2f}, Recall_ratio:{:.2f}, ' \
                     'Pred_inlier:{:f}, Gt_inlier:{:f}'.format(self.epoch,
                                                               np.mean(mean_loss),
                                                               np.mean(mean_inlier_ratio),
                                                               np.mean(mean_recall_ratio),
                                                               np.mean(mean_corr_matches),
                                                               np.mean(mean_gt_matches),
                                                               )
        print(print_text)
        self.log_file.write('\n' + print_text + '\n')

        return np.mean(mean_inlier_ratio)

    def eval_matching(self, epoch=0):
        self.model.eval()
        with open(self.args.eval_config, 'rt') as f:
            # t_args.__dict__.update(json.load(f))
            # opt = eval_parser.parse_args(namespace=t_args)
            opt = json.load(f)
            opt['output_dir'] = osp.join(self.save_dir, 'vis_eval_epoch_{:02d}'.format(epoch))
            # print(opt)
            opt['feature'] = self.args.feature

        with torch.no_grad():
            # eval_out = evaluate(superglue=self.model, superpoint=None, opt=opt)
            # eval_out = evaluate_graph_matcher(net=self.model, opt=opt)
            for dataset in ['scannet', 'yfcc']:
                eval_out = evaluate_full(model=self.model, opt=opt, dataset=dataset, feat_type=self.args.feature)

                for k, v in eval_out.items():
                    self.writer.add_scalar(tag=dataset + '_eval_' + k, scalar_value=v, global_step=self.iteration)

                text = "Eval Epoch [{:d}] for {:s}".format(epoch, dataset)
                for k in eval_out.keys():
                    text = text + " {:s} [{:.2f}]".format(k, eval_out[k])
                self.log_file.write(text + "\n\n")
                self.log_file.flush()

                # torch.cuda.empty_cache()

        return eval_out['prec']

    def train(self):
        if self.args.local_rank == 0:
            print('Start to train the model from epoch: {:d}'.format(self.epoch))
            hist_values = []
            min_value = self.min_loss

        epoch = self.epoch

        while epoch < self.num_epochs:
            if self.args.with_dist > 0:
                self.train_loader.sampler.set_epoch(epoch=epoch)
            self.epoch = epoch

            train_loss = self.process_epoch()

            # return with loss INF/NAN
            if train_loss is None:
                continue

            if self.args.local_rank == 0:
                if self.do_eval and self.epoch % 5 == 0:  # and self.epoch >= 50:
                    eval_ratio = self.eval_fun(epoch=self.epoch)

                    hist_values.append(eval_ratio)  # higher better
                else:
                    hist_values.append(-train_loss)  # lower better

                checkpoint_path = os.path.join(
                    self.save_dir,
                    '%s.%02d.pth' % (self.args.network, self.epoch)
                )
                checkpoint = {
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'model': self.model.state_dict(),
                    # 'optimizer': self.optimizer.state_dict(),
                    # 'scheduler': self.lr_scheduler.state_dict(),
                    'min_loss': min_value,
                }
                # for multi-gpu training
                if len(self.args.gpu) > 1:
                    checkpoint['model'] = self.model.module.state_dict()

                torch.save(checkpoint, checkpoint_path)

                if hist_values[-1] < min_value:
                    min_value = hist_values[-1]
                    best_checkpoint_path = os.path.join(
                        self.save_dir,
                        '%s.best.pth' % (self.tag)
                    )
                    shutil.copy(checkpoint_path, best_checkpoint_path)
            # important!!!
            epoch += 1
            # self.lr_scheduler.step()
            if self.args.dataset in ('megadepth', 'mscoco'):
                self.train_loader.dataset.build_dataset(seed=self.epoch)

        if self.args.local_rank == 0:
            self.log_file.close()
