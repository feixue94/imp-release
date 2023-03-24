import torch
import torch.distributed as dist
import numpy as np
import cv2

def parse_pair_seq(pair_num_list):
    #generate pair_seq_list: [#pair_num]:seq
    #              accu_pair_num: dict{seq_name:accumulated_pair}
    pair_num=int(pair_num_list[0,1])
    pair_num_list=pair_num_list[1:]
    pair_seq_list=[]
    cursor=0
    accu_pair_num={}
    for line in pair_num_list:
       seq,seq_pair_num=line[0],int(line[1])
       for _ in range(seq_pair_num):
          pair_seq_list.append(seq)
       accu_pair_num[seq]=cursor
       cursor+=seq_pair_num
    assert pair_num==cursor
    return pair_seq_list,accu_pair_num

def tocuda(data):
    # convert tensor data in dictionary to cuda when it is a tensor
    for key in data.keys():
        if type(data[key]) == torch.Tensor:
            data[key] = data[key].cuda()
    return data
    
def reduce_tensor(tensor,op='mean'): 
    rt = tensor.detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if op=='mean':
        rt /= dist.get_world_size()
    return rt

def get_rnd_homography(batch_size, pert_ratio=0.25):
    corners = np.array([[-1, 1], [1, 1], [-1, -1], [1, -1]], dtype=np.float32)
    homo_tower = []
    for _ in range(batch_size):
        rnd_pert = np.random.uniform(-2 * pert_ratio, 2 * pert_ratio, (4, 2)).astype(np.float32)
        pert_corners = corners + rnd_pert
        M = cv2.getPerspectiveTransform(corners, pert_corners)
        homo_tower.append(M)
    homo_tower = np.stack(homo_tower, axis=0)

    return homo_tower
