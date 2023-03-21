import os
import os.path as osp
import functools
import cv2
import yaml
import numpy as np
from tqdm import trange, tqdm
import torch
import torch.utils.data as Data
from torch.autograd import Variable

from torch.multiprocessing import Process, Manager, set_start_method, Pool

from components.readers import standard_reader, reader_set
from components.evaluators import auc_eval, FMbench_eval
from utils import evaluation_utils, metrics
from dataset_loc.megadepth import Megadepth_loc


# from utils import evaluation_utils, metrics
def feed_match(info, matcher):
    x1, x2, desc1, desc2, size1, size2 = info['x1'], info['x2'], info['desc1'], info['desc2'], info['img1'].shape[:2], \
                                         info['img2'].shape[:2]
    test_data = {'x1': x1, 'x2': x2, 'desc1': desc1, 'desc2': desc2, 'size1': np.flip(np.asarray(size1)),
                 'size2': np.flip(np.asarray(size2))}
    corr1, corr2 = matcher.run(test_data)
    return [corr1, corr2]


def feed_match_v2(info, model, p_th=0.2):
    with torch.no_grad():
        def match_p(p):  # p N*M
            score, index = torch.topk(p, k=1, dim=-1)
            _, index2 = torch.topk(p, k=1, dim=-2)
            mask_th, index, index2 = score[:, 0] > p_th, index[:, 0], index2.squeeze(0)
            mask_mc = index2[index] == torch.arange(len(p)).cuda()
            mask = mask_th & mask_mc
            index1, index2 = torch.nonzero(mask).squeeze(1), index[mask]
            return index1, index2

        x1, x2, desc1, desc2, size1, size2 = info['x1'], info['x2'], info['desc1'], info['desc2'], info['img1'].shape[
                                                                                                   :2], \
                                             info['img2'].shape[:2]
        test_data = {'x1': x1, 'x2': x2, 'desc1': desc1, 'desc2': desc2, 'size1': np.flip(np.asarray(size1)),
                     'size2': np.flip(np.asarray(size2))}
        norm_x1, norm_x2 = evaluation_utils.normalize_size(test_data['x1'][:, :2], test_data['size1'], scale=0.7), \
                           evaluation_utils.normalize_size(test_data['x2'][:, :2], test_data['size2'], scale=0.7)

        # print('x1: ', test_data['x1'][:, :2])
        # print('norm_x1: ', norm_x1)
        # print('size1: ', test_data['size1'])
        norm_x1, norm_x2 = np.concatenate([norm_x1, test_data['x1'][:, 2, np.newaxis]], axis=-1), \
                           np.concatenate([norm_x2, test_data['x2'][:, 2, np.newaxis]], axis=-1)
        feed_data = {'x1': torch.from_numpy(norm_x1[np.newaxis]).cuda().float(),
                     'x2': torch.from_numpy(norm_x2[np.newaxis]).cuda().float(),
                     'desc1': torch.from_numpy(test_data['desc1'][np.newaxis]).cuda().float(),
                     'desc2': torch.from_numpy(test_data['desc2'][np.newaxis]).cuda().float(),
                     'keypoints1': torch.from_numpy(info['x1'][np.newaxis]).cuda().float(),
                     'keypoints2': torch.from_numpy(info['x2'][np.newaxis]).cuda().float(),
                     'intrinsics1': torch.from_numpy(info['K1'][np.newaxis]).cuda().float(),
                     'intrinsics2': torch.from_numpy(info['K2'][np.newaxis]).cuda().float(),
                     }

        res = model(data=feed_data, mode=1)
        if 'p' in res.keys():
            p = res['p']
            index1, index2 = match_p(p[0, :-1, :-1])
        else:
            index1 = res['index0']
            index2 = res['index1']
        corr1, corr2 = test_data['x1'][:, :2][index1.cpu()], test_data['x2'][:, :2][index2.cpu()]
        if len(corr1.shape) == 1:
            corr1, corr2 = corr1[np.newaxis], corr2[np.newaxis]

        out = {
            'corr1': corr1,
            'corr2': corr2,
        }
        if 'pred_pose' in res.keys():
            out['pred_pose'] = res['pred_pose']
        # return [corr1, corr2]
        return out


def reader_handler(reader, read_que):
    # reader = load_component('reader', config['name'], config)
    for index in range(len(reader)):
        index += 0
        info = reader.run(index)
        read_que.put(info)
    read_que.put('over')


def evaluate_full(model, opt, feat_type='spp', dataset='yfcc', max_length=None, max_keypoints=-1):
    if feat_type == 'spp':
        if dataset == 'yfcc':
            config_path = 'configs/yfcc_eval_gm.yaml'
        elif dataset == 'scannet':
            config_path = 'configs/scannet_eval_gm.yaml'
        elif dataset == 'fm':
            config_path = 'configs/fm_eval_gm.yaml'
    else:
        if dataset == 'yfcc':
            config_path = 'configs/yfcc_eval_gm_sift.yaml'
        elif dataset == 'scannet':
            config_path = 'configs/scannet_eval_gm_sift.yaml'
        elif dataset == 'fm':
            config_path = 'configs/fm_eval_gm_sift.yaml'

    if dataset == 'yfcc':
        th = 1
        inlier_th = 0.005
    elif dataset == 'scannet':
        th = 3
        inlier_th = 0.005
    else:
        th = 1
        inlier_th = 0.003

    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.Loader)
        read_config = config['reader']
        eval_config = config['evaluator']

    if opt is not None:
        vis_folder = str(opt['output_dir'])
        if vis_folder is not None:
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder, exist_ok=True)

    if max_keypoints > 0:
        read_config['num_kpt'] = max_keypoints
    # reader = load_component('reader', read_config['name'], read_config)
    reader = standard_reader(config=read_config)
    reader_loader = Data.DataLoader(dataset=reader, num_workers=4, shuffle=False)
    matcher = feed_match_v2
    # evaluator = load_component('evaluator', eval_config['name'], eval_config)

    if dataset == 'fm':
        evaluator = FMbench_eval(config=eval_config)
    else:
        evaluator = auc_eval(config=eval_config)

    # results = {}
    for index in tqdm(range(len(reader_loader)), total=len(reader)):
        # index += 0
        if max_length is not None:
            if index >= max_length:
                break
        info = reader.run(index)
        # corr1, corr2 = matcher(info=info, model=model, p_th=0.2)
        # cur_res = evaluator.run({**info, **{'corr1': corr1, 'corr2': corr2}}, th=th)

        match_out = matcher(info=info, model=model, p_th=0.2)
        if 'pred_pose' in match_out.keys():
            if match_out['pred_pose'] is not None:
                match_out['pred_pose'] = match_out['pred_pose'][0].cpu().numpy()
            else:
                match_out['pred_pose'] = np.eye(4, dtype=float)
        cur_res = evaluator.run({**info, **match_out}, th=th)

        evaluator.res_inqueue(res=cur_res)

        # break
        # if index >= 20:
        #     break

    reader.close()
    output = evaluator.parse()
    aucs = output['exact_auc']
    prec = output['mean_precision']
    mscore = output['mean_match_score']

    # print('Evaluation Results (mean over {} pairs):'.format(len(reader)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0] * 100, aucs[1] * 100, aucs[3] * 100, prec, mscore))

    result = {
        "auc@5": aucs[0] * 100,
        "auc@10": aucs[1] * 100,
        "auc@15": aucs[2] * 100,
        "auc@20": aucs[3] * 100,
        "prec": prec,
        "mscore": mscore,
    }

    if 'exact_auc_w8' in output.keys():
        aucs_w8 = output['exact_auc_w8']
        print('W8AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs_w8[0] * 100, aucs_w8[1] * 100, aucs_w8[3] * 100, prec, mscore))

        result['auc@5_w8'] = aucs_w8[0] * 100
        result['auc@10_w8'] = aucs_w8[1] * 100
        result['auc@15_w8'] = aucs_w8[2] * 100
        result['auc@20_w8'] = aucs_w8[3] * 100

    return result


def match_handler(matcher, read_que, match_que):
    # matcher = load_component('matcher', config['name'], config)
    # match_func = functools.partial(feed_match, matcher=matcher)
    match_func = functools.partial(feed_match_v2, model=matcher)
    pool = Pool(4)
    cache = []
    while True:
        item = read_que.get()
        # clear cache
        if item == 'over':
            if len(cache) != 0:
                results = pool.map(match_func, cache)
                for cur_item, cur_result in zip(cache, results):
                    cur_item['corr1'], cur_item['corr2'] = cur_result[0], cur_result[1]
                    match_que.put(cur_item)
            match_que.put('over')
            break
        cache.append(item)
        # print(len(cache))
        if len(cache) == 4:
            # matching in parallel
            results = pool.map(match_func, cache)
            for cur_item, cur_result in zip(cache, results):
                cur_item['corr1'], cur_item['corr2'] = cur_result[0], cur_result[1]
                match_que.put(cur_item)
            cache = []
    pool.close()
    pool.join()


def evaluate_handler(evaluator, match_que, num_pair):
    # evaluator = load_component('evaluator', config['name'], config)
    pool = Pool(4)
    cache = []
    for _ in trange(num_pair):
        item = match_que.get()
        if item == 'over':
            if len(cache) != 0:
                results = pool.map(evaluator.run, cache)
                for cur_res in results:
                    evaluator.res_inqueue(cur_res)
            break
        cache.append(item)
        if len(cache) == 4:
            results = pool.map(evaluator.run, cache)
            for cur_res in results:
                evaluator.res_inqueue(cur_res)
            cache = []
        # if args.vis_folder is not None:
        # dump visualization
        # corr1_norm, corr2_norm = evaluation_utils.normalize_intrinsic(item['corr1'], item['K1']), \
        #                          evaluation_utils.normalize_intrinsic(item['corr2'], item['K2'])
        # inlier_mask = metrics.compute_epi_inlier(corr1_norm, corr2_norm, item['e'], config['inlier_th'])
        # display = evaluation_utils.draw_match(item['img1'], item['img2'], item['corr1'], item['corr2'], inlier_mask)
        # cv2.imwrite(os.path.join(args.vis_folder, str(item['index']) + '.png'), display)
    evaluator.parse()


def evaluate_full_multiprocess(model, opt, feat_type='spp', dataset='yfcc'):
    if feat_type == 'spp':
        if dataset == 'yfcc':
            config_path = 'configs/yfcc_eval_gm.yaml'
        elif dataset == 'scannet':
            config_path = 'configs/scannet_eval_gm.yaml'
        elif dataset == 'fm':
            config_path = 'configs/fm_eval_gm.yaml'
    else:
        if dataset == 'yfcc':
            config_path = 'configs/yfcc_eval_gm_sift.yaml'
        elif dataset == 'scannet':
            config_path = 'configs/scannet_eval_gm_sift.yaml'
        elif dataset == 'fm':
            config_path = 'configs/fm_eval_gm_sift.yaml'

    if dataset == 'yfcc':
        th = 1
        inlier_th = 0.005
    elif dataset == 'scannet':
        th = 3
        inlier_th = 0.005
    else:
        th = 1
        inlier_th = 0.003

    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.Loader)
        read_config = config['reader']
        eval_config = config['evaluator']

    if opt is not None:
        vis_folder = str(opt['output_dir'])
        if vis_folder is not None and not os.path.exists(vis_folder):
            os.mkdir(vis_folder)

    # reader = load_component('reader', read_config['name'], read_config)
    reader = standard_reader(config=read_config)
    # matcher = feed_match_v2

    if dataset == 'fm':
        evaluator = FMbench_eval(config=eval_config)
    else:
        evaluator = auc_eval(config=eval_config)

    read_que, match_que, estimate_que = Manager().Queue(maxsize=100), Manager().Queue(maxsize=100), Manager().Queue(
        maxsize=100)
    read_process = Process(target=reader_handler, args=(reader, read_que))
    match_process = Process(target=match_handler, args=(model, read_que, match_que))
    eval_process = Process(target=evaluate_handler, args=(evaluator, match_que, eval_config['num_pair']))

    read_process.start()
    match_process.start()
    eval_process.start()

    read_process.join()
    match_process.join()
    eval_process.join()


def evaluate_mega(model, args, feat_type='spp', dataset='mega'):
    val_set = Megadepth_loc(
        scene_info_path=osp.join(args.data_root, args.scene_info_path),
        base_path=osp.join(args.data_root, args.base_path),
        scene_list_fn=args.scene_list_fn_validation,
        min_overlap_ratio=args.min_overlap_ratio,
        max_overlap_ratio=args.max_overlap_ratio,
        pairs_per_scene=args.pairs_per_scene,
        image_size=args.image_size,
        nfeatures=args.max_keypoints,
        train=False,
        # inlier_th=args.inlier_th,
        feature_type=args.feature,
        pre_load_scene_info=(args.pre_load > 0),
        min_inliers=args.min_inliers,
        max_inliers=args.max_inliers,
        random_inliers=(args.random_inliers > 0),
        matches={},  # matches,
    )
    val_set.build_dataset(seed=0)
    val_loader = torch.utils.data.DataLoader(val_set, num_workers=8, shuffle=False, batch_size=1)

    all_errs_cv = []
    all_errs = []
    all_mscores = []
    all_precisions = []
    for bid, pred in tqdm(enumerate(val_loader), total=len(val_loader)):
        for k in pred:
            if k.find('file_name') >= 0:
                continue
            if k != 'image0' and k != 'image1' and k != 'depth0' and k != 'depth1':
                if type(pred[k]) == torch.Tensor:
                    pred[k] = Variable(pred[k].float().cuda())
                else:
                    pred[k] = Variable(torch.stack(pred[k]).float().cuda())

        # print('norm_keypoints0: ', pred['norm_keypoints0'])
        # print('norm_keypoints1: ', pred['norm_keypoints1'])
        with torch.no_grad():
            out = model(pred)

            pred_matches = out['indices0'][-1]
            pred_mscores = out['mscores0'][-1]

            gt_matches = pred['matching_mask'][:, :-1, :].max(-1).indices
            gt_matches[gt_matches == pred['matching_mask'].shape[2] - 1] = -1
            corr = torch.abs(pred_matches - gt_matches) == 0
            valid = (gt_matches >= 0)
            prec = torch.sum(corr[valid]) / torch.sum(valid)
            all_precisions.append(prec.cpu().numpy())
            all_mscores.append(torch.mean(pred_mscores[pred_matches >= 0]).cpu().numpy())

        gt_P = pred['gt_P'][0].cpu().numpy()

        if 'pred_pose' in out.keys():
            if out['pred_pose'] is not None:
                pred_pose = out['pred_pose'][0].cpu().numpy()
                err_R, err_t = metrics.evaluate_R_t(R_gt=gt_P[:3, :3], t_gt=gt_P[:3, 3],
                                                    R=pred_pose[:3, :3], t=pred_pose[:3, 3])

                all_errs.append(max([err_R, err_t]))
            else:
                all_errs.append(100)

        else:
            all_errs.append(100)

        pred_matches = pred_matches[0]
        matching_mask = (pred_matches >= 0)
        p3ds_mask = pred['points3d_mask'][0][pred_matches]
        mask = torch.logical_and(matching_mask, p3ds_mask)

        matched_kpts = pred['keypoints0'][0][mask].cpu().numpy()
        matched_p3ds = pred['points3d_world'][0][pred_matches][mask].cpu().numpy()

        K = pred['K'][0].cpu().numpy()

        if matched_kpts.shape[0] < 4:
            found0 = False
        else:
            found0, rvec0, tvec0, inliers0 = cv2.solvePnPRansac(objectPoints=matched_p3ds,
                                                                imagePoints=matched_kpts,
                                                                cameraMatrix=K,
                                                                distCoeffs=None,
                                                                iterationsCount=1000)

        if found0:
            R0, _ = cv2.Rodrigues(rvec0)

            pred_P0 = np.eye(4)
            pred_P0[:3, :3] = R0
            pred_P0[:3, 3] = tvec0.reshape(-1)
            err_R_cv, err_t_cv = metrics.evaluate_R_t(R_gt=gt_P[:3, :3], t_gt=gt_P[:3, 3], R=R0, t=tvec0)
            all_errs_cv.append(np.max([err_R_cv, err_t_cv]))
            # print('pred_R0: ', R0)
            # print('pred_t0: ', tvec0)
            # print('err: ', err_R_cv, err_t_cv)
        else:
            all_errs_cv.append(100)

    ths = np.arange(7) * 5
    aucs = metrics.pose_auc(errors=all_errs, thresholds=ths)
    aucs_cv = metrics.pose_auc(errors=all_errs_cv, thresholds=ths)
    prec = np.mean(all_precisions)
    mscore = np.mean(all_mscores)

    result = {
        "auc@5": aucs[0] * 100,
        "auc@10": aucs[1] * 100,
        "auc@15": aucs[2] * 100,
        "auc@20": aucs[3] * 100,
        "auc@5_cv": aucs_cv[0] * 100,
        "auc@10_cv": aucs_cv[1] * 100,
        "auc@15_cv": aucs_cv[2] * 100,
        "auc@20_cv": aucs_cv[3] * 100,
        "prec": prec,
        "mscore": mscore,
    }
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0] * 100, aucs[1] * 100, aucs[3] * 100, prec, mscore))
    print('cv: AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs_cv[0] * 100, aucs_cv[1] * 100, aucs_cv[3] * 100, prec, mscore))

    return result
