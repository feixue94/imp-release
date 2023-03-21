# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp-release -> train
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   21/03/2023 21:45
=================================================='''

# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pnba -> train
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/12/2021 14:22
=================================================='''

import argparse
import json
import os
import os.path as osp
import torch
import torchvision.transforms.transforms as tvf
import torch.utils.data as Data
from nets.superglue import SuperGlue
from nets.graphmatching import GraphMatcher
from nets.gm import GM
from nets.gmb import GMB
from nets.gmn import GMN
from nets.gmn import DGNN, DGNNP, DGNNS, DGNNPS
from nets.spgo import SPGO
from nets.adgm import AdaGMN
from trainer import Trainer
from dataset.mscoco import MSCOCO
from dataset.megadepth import Megadepth, read_matches
from tools.common import torch_set_gpu
import torch.multiprocessing as mp
import torch.distributed as dist

torch.set_grad_enabled(True)

parser = argparse.ArgumentParser(description='Superglue', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vis', action='store_true', help='visualization of matches')
parser.add_argument('--eval', action='store_true', help='evaluation')
parser.add_argument('--max_keypoints', type=int, default=512, help='the maximum number of keypoints')
parser.add_argument('--keypoint_th', type=float, default=0.005, help='threshold of superpoint detector')
parser.add_argument('-nms_radius', type=int, default=3, help='nms radius')
parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                    help='the number of Sinkhorn iterations in Superglue')
parser.add_argument('--match_th', type=float, default=0.2, help='Superglue matching threshold')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--dataset', type=str, default='mscoco', choices={'mscoco', 'megadepth'})
parser.add_argument('--feature', choices={'sift', 'spp'}, default='spp', help='features used for training')
parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='-1 for CPU')
parser.add_argument("--its_per_epoch", type=int, default=-1)
parser.add_argument("--log_intervals", type=int, default=50)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--save_root', type=str, default='/scratches/flyer_3/fx221/exp/pnba')
parser.add_argument('--data_root', type=str, default='/scratches/flyer_2/fx221/superglue/train2014')
parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='indoor', help='SuperGlue weights')
parser.add_argument('--network', type=str, default='superglue', help='SuperGlue weights')
parser.add_argument('--config', type=str, required=True, help='config of specifications')
parser.add_argument('--eval_config', type=str, default=None, help='config of specifications')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_DDP(rank, world_size, model, args):
    print('In train_DDP..., rank: ', rank)
    torch.cuda.set_device(rank)

    if args.dataset == 'megadepth':
        train_set = Megadepth(
            scene_info_path=osp.join(args.data_root, args.scene_info_path),
            base_path=osp.join(args.data_root, args.base_path),
            scene_list_fn=args.scene_list_fn,
            min_overlap_ratio=args.min_overlap_ratio,
            max_overlap_ratio=args.max_overlap_ratio,
            pairs_per_scene=args.pairs_per_scene,
            image_size=args.image_size,
            nfeatures=args.max_keypoints,
            train=(args.train > 0),
            inlier_th=args.inlier_th,
            feature_type=args.feature,
            pre_load_scene_info=(args.pre_load > 0),
            extract_feature=False,
            min_inliers=args.min_inliers,
            max_inliers=args.max_inliers,
            random_inliers=(args.random_inliers > 0),
            # tmp_dir='/scratches/flyer_3/fx221/exp/pnba/megadepth_spp',
            matches={},
            rank=rank,
        )
    device = torch.device(f'cuda:{rank}')
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    setup(rank=rank, world_size=world_size)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=False, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size // world_size,
                                               num_workers=args.workers // world_size,
                                               pin_memory=False,
                                               sampler=train_sampler)
    args.local_rank = rank
    trainer = Trainer(model=model, train_loader=train_loader, eval_loader=None, args=args)
    trainer.train()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')  # don't use for mega [keypoint, matches], will cause ram oom
    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    torch_set_gpu(gpus=args.gpu)
    # args.save_root = osp.join(args.data_root, args.save_root)
    args.save_root = osp.join('/scratches/flyer_3', args.save_root)
    if args.local_rank == 0:
        print(args)

    config = {
        'superpoint': {
            'nms_radius': args.nms_radius,
            'keypoint_threshold': args.keypoint_th,
            'max_keypoints': args.max_keypoints
        },
        'matcher': {
            'sinkhorn_iterations': args.sinkhorn_iterations,
            'match_threshold': args.match_th,
            'descriptor_dim': 256 if args.feature == 'spp' else 128,
            'GNN_layers': ['self', 'cross'] * args.layers,
            'n_layers': args.layers,
            'ac_fn': args.ac_fn,
            'with_sinkhorn': (args.with_sinkhorn > 0),

            # for adaptive pooling
            'n_min_tokens': args.n_min_tokens,

            # for pose estimator
            'with_pose': (args.with_pose > 0),
            'n_hypothesis': args.n_hypothesis,
            'error_th': args.error_th,
            'inlier_th': args.inlier_th,
            'pose_type': 'H' if args.dataset == 'mscoco' else 'F',
            'minium_samples': 4 if args.dataset == 'mscoco' else 8,
        }
    }

    if args.network == 'superglue':
        model = SuperGlue(config.get('matcher', {}))
    elif args.network == 'graphmatcher':
        model = GraphMatcher(config.get('matcher', {}))
    elif args.network == 'gm':
        model = GM(config.get('matcher', {}))
    elif args.network == 'dgnn':
        model = DGNN(config.get('matcher', {}))
    elif args.network == 'dgnnp':
        model = DGNNP(config.get('matcher', {}))
    elif args.network == 'dgnns':
        model = DGNNS(config.get('matcher', {}))
    elif args.network == 'dgnnps':
        model = DGNNPS(config.get('matcher', {}))

    if args.local_rank == 0:
        print('model: ', model)
        # load pretrained weight
        if args.weight_path != "None":
            model.load_state_dict(torch.load(osp.join(args.save_root, args.weight_path), map_location='cpu')['model'],
                                  strict=True)
            print('Load weight from {:s}'.format(osp.join(args.save_root, args.weight_path)))

        if args.resume_path != 'None':
            model.load_state_dict(torch.load(osp.join(args.save_root, args.resume_path), map_location='cpu')['model'],
                                  strict=True)
            print('Load resume weight from {:s}'.format(osp.join(args.save_root, args.resume_path)))

    if args.with_dist < 1:
        if args.dataset == 'megadepth':
            base_path = '/scratch/fx221/dataset/Megadepth'
            scenes = []
            with open(args.scene_list_fn, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    scenes.append(l.strip())
            matches = read_matches(scenes=scenes, base_path=base_path)

            train_set = Megadepth(
                scene_info_path=osp.join(args.data_root, args.scene_info_path),
                base_path=osp.join(args.data_root, args.base_path),
                scene_list_fn=args.scene_list_fn,
                min_overlap_ratio=args.min_overlap_ratio,
                max_overlap_ratio=args.max_overlap_ratio,
                pairs_per_scene=args.pairs_per_scene,
                image_size=args.image_size,
                nfeatures=args.max_keypoints,
                train=(args.train > 0),
                inlier_th=args.inlier_th,
                feature_type=args.feature,
                pre_load_scene_info=(args.pre_load > 0),
                extract_feature=False,
                min_inliers=args.min_inliers,
                max_inliers=args.max_inliers,
                random_inliers=(args.random_inliers > 0),
                # tmp_dir='/scratches/flyer_3/fx221/exp/pnba/megadepth_spp',
                local_rank=args.local_rank,
                matches=matches,
            )
            eval_set = None

        train_loader = Data.DataLoader(dataset=train_set,
                                       shuffle=False,
                                       batch_size=args.batch_size,
                                       drop_last=True,
                                       num_workers=args.workers)
        model = model.cuda()
        trainer = Trainer(model=model, train_loader=train_loader, eval_loader=None, args=args)
        trainer.train()
    else:
        mp.spawn(train_DDP, nprocs=len(args.gpu), args=(len(args.gpu), model, args), join=True)
