# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   imp-release -> train
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   21/03/2023 21:45
=================================================='''
import argparse
import json
import os
import os.path as osp
import torch
import torch.utils.data as Data
from nets.gm import GM
from nets.adgm import AdaGMN
from nets.gms import DGNNS
from trainer import Trainer
from dataset.megadepth import Megadepth
from tools.common import torch_set_gpu
import torch.multiprocessing as mp
import torch.distributed as dist

torch.set_grad_enabled(True)

parser = argparse.ArgumentParser(description='IMP', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eval', action='store_true', help='evaluation')
parser.add_argument('--max_keypoints', type=int, default=512, help='the maximum number of keypoints')
parser.add_argument('--keypoint_th', type=float, default=0.005, help='threshold of superpoint detector')
parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                    help='the number of Sinkhorn iterations in Superglue')
parser.add_argument('--match_th', type=float, default=0.2, help='Superglue matching threshold')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--feature', choices={'sift', 'spp'}, default='spp', help='features used for training')
parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='-1 for CPU')
parser.add_argument("--its_per_epoch", type=int, default=-1)
parser.add_argument("--log_intervals", type=int, default=50)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--save_path', type=str, default='/scratches/flyer_3/fx221/exp/imp')
parser.add_argument('--base_path', type=str, default='/scratches/flyer_3/fx221/dataset/Megadepth')
parser.add_argument('--network', type=str, default='gm')
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
    train_set = Megadepth(
        scene_info_path=osp.join(args.base_path, 'scene_info'),
        base_path=args.base_path,
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
    # torch.multiprocessing.set_start_method('spawn')  # don't use for mega [keypoint, matches], cause oom
    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    torch_set_gpu(gpus=args.gpu)
    if args.local_rank == 0:
        print(args)

    config = {
        'sinkhorn_iterations': args.sinkhorn_iterations,
        'match_threshold': args.match_th,
        'descriptor_dim': 256 if args.feature == 'spp' else 128,
        'GNN_layers': ['self', 'cross'] * args.layers,
        'n_layers': args.layers,
        'ac_fn': args.ac_fn,
        'norm_fn': args.norm_fn,
        'with_sinkhorn': (args.with_sinkhorn > 0),

        # for adaptive pooling
        'n_min_tokens': args.n_min_tokens,

    }

    if args.network == 'gm':
        model = GM(config)
    elif args.network == 'dgnns':
        model = DGNNS(config)
    elif args.network == 'adagmn':
        model = AdaGMN(config)

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

    mp.spawn(train_DDP, nprocs=len(args.gpu), args=(len(args.gpu), model, args), join=True)
