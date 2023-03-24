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
                self.args.layers) + '_' + str(self.args.feature) + '_B' + str(
                self.args.batch_size) + '_K' + str(self.args.max_keypoints) + '_M' + str(
                self.args.match_th) + '_' + self.args.ac_fn + '_' + self.args.norm_fn + '_' + self.args.optim

        self.save_dir = osp.join(self.args.save_path, log_dir)
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        print("save_dir: ", self.save_dir)

        self.log_file = open(osp.join(self.save_dir, "log.txt"), "a+")
        if self.args.local_rank == 0:
            save_args(args=args, save_path=Path(self.save_dir, "args.txt"))
        self.writer = SummaryWriter(self.save_dir)

        self.tag = log_dir

        self.do_eval = (self.args.do_eval > 0)
        if self.do_eval:
            self.eval_fun = self.eval_matching

    def process_epoch(self):
        self.model.train()

        epoch_losses = []
        epoch_acc_corr = []
        epoch_acc_incorr = []
        epoch_acc_corr_ratio = []
        epoch_acc_incorr_ratio = []
        epoch_matching_loss = []

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

            if self.args.its_per_epoch >= 0 and bidx >= self.args.its_per_epoch:
                break

            data = self.model(pred)
            for k, v in pred.items():
                pred[k] = v
            pred = {**pred, **data}

            loss = pred['loss']
            acc_corr = pred['acc_corr'][-1]
            acc_incorr = pred['acc_incorr'][-1]
            total_acc_corr = pred['total_acc_corr'][-1]
            total_acc_incorr = pred['total_acc_incorr'][-1]
            if 'matching_loss' in pred.keys():
                matching_loss = pred['matching_loss']
            else:
                matching_loss = loss

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
                print_text = 'Epoch [{:d}/{:d}], Step [{:d}/{:d}/{:d}], Loss [m{:.2f}//t{:.2f}], MS [{:.2f}], Acc [c{:.1f}/{:.1f}, n{:.1f}/{:.1f}]'.format(
                    self.epoch,
                    self.num_epochs, bidx,
                    len(self.train_loader),
                    self.iteration,
                    matching_loss.item(),
                    loss.item(),
                    torch.max(matching_score).item(),
                    np.mean(epoch_acc_corr),
                    np.mean(epoch_acc_corr_ratio),
                    np.mean(epoch_acc_incorr),
                    np.mean(epoch_acc_incorr_ratio),
                )
                print(print_text)
                self.log_file.write(print_text + '\n')

                info = {
                    'lr': lr,
                    'matching_loss': matching_loss.item(),
                    'loss': loss.item(),
                    'acc_corr': acc_corr.item(),
                    'acc_incorr': acc_incorr.item(),
                    'acc_corr_ratio': acc_corr_ratio,
                    'acc_incorr_ratio': acc_incorr_ratio,
                }
                for k, v in info.items():
                    self.writer.add_scalar(tag=k, scalar_value=v, global_step=self.iteration)

        if self.args.local_rank == 0:
            print_text = 'Epoch [{:d}/{:d}], AVG Loss [m{:.2f}/t{:.2f}], Acc [c{:.1f}/{:.1f}, n{:.1f}/{:.1f}]\n'.format(
                self.epoch,
                self.num_epochs,
                np.mean(epoch_matching_loss),
                np.mean(epoch_losses),
                np.mean(epoch_acc_corr),
                np.mean(epoch_acc_corr_ratio),
                np.mean(epoch_acc_incorr),
                np.mean(epoch_acc_incorr_ratio),
            )
            print(print_text)
            self.log_file.write(print_text + '\n')
            self.log_file.flush()
        return np.mean(epoch_losses)

    def eval_matching(self, epoch=0):
        self.model.eval()
        with open(self.args.eval_config, 'rt') as f:
            opt = json.load(f)
            opt['output_dir'] = osp.join(self.save_dir, 'vis_eval_epoch_{:02d}'.format(epoch))
            opt['feature'] = self.args.feature

        with torch.no_grad():
            for dataset in ['scannet', 'yfcc']:
                eval_out = evaluate_full(model=self.model, opt=opt, dataset=dataset, feat_type=self.args.feature)

                for k, v in eval_out.items():
                    self.writer.add_scalar(tag=dataset + '_eval_' + k, scalar_value=v, global_step=self.iteration)

                text = "Eval Epoch [{:d}] for {:s}".format(epoch, dataset)
                for k in eval_out.keys():
                    text = text + " {:s} [{:.2f}]".format(k, eval_out[k])
                self.log_file.write(text + "\n\n")
                self.log_file.flush()

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
            self.train_loader.dataset.build_dataset(seed=self.epoch)

        if self.args.local_rank == 0:
            self.log_file.close()
