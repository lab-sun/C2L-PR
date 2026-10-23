import os
import sys
import argparse

# Read --gpu before importing torch, which reads CUDA_VISIBLE_DEVICES once.
# Without --gpu, the script keeps the caller's CUDA_VISIBLE_DEVICES.
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument('--gpu', type=str, default=None)
_gpu_args, _ = _gpu_parser.parse_known_args()
if _gpu_args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_args.gpu

import warnings

import torch
from net import RINet_attention
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
# import matplotlib
# matplotlib.use('TkAgg')
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inverse_squared_l2(desc1, desc2):
    """Per-row 1 / ||desc1 - desc2||^2. A zero distance is +inf.

    The sum is float64 so NumPy 1.x and NumPy 2 (NEP 50) agree, and so a
    large but finite bin does not overflow to a similarity of 0.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        diff = np.asarray(desc1, dtype=np.float64) - np.asarray(desc2, dtype=np.float64)
        sq = np.sum(diff * diff, axis=1)
        return 1.0 / sq


def max_inverse_squared_l2(desc1, desc2_list):
    """Best-view inverse squared L2.

    NaN views are ignored. A row that is NaN on every view becomes 0.
    +inf (a perfect match) becomes the float64 maximum.
    """
    cols = []
    for desc2 in desc2_list:
        cols.append(np.asarray(inverse_squared_l2(desc1, desc2)).reshape(-1, 1))
    stacked = np.concatenate(cols, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # np.max would turn one NaN view into a score of 0 and drop a real match.
        best = np.nanmax(stacked, axis=1)
    return np.nan_to_num(best)


def f1_from_precision_recall(precision, recall):
    """Harmonic mean of precision and recall. 0/0 is 0, not NaN."""
    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / (precision + recall)
    return np.nan_to_num(f1)


def max_f1_from_pr(precision, recall):
    return np.max(f1_from_precision_recall(precision, recall))


def pr_max_f1(y_true, y_score):
    """Maximum F1 along the precision-recall curve. Higher scores are better."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, _thresholds = metrics.precision_recall_curve(
            np.asarray(y_true).reshape(-1),
            np.asarray(y_score, dtype=np.float64).reshape(-1),
        )
    return max_f1_from_pr(precision, recall)


def max_view_scores(score_list):
    """Elementwise max across score vectors, one vector per view."""
    cols = [np.asarray(score).reshape(-1, 1) for score in score_list]
    return np.max(np.concatenate(cols, axis=1), axis=1)


def fast_eval(seq='00', model_file="./model/attention/00.ckpt", pair_file='./data/pairs_kitti/neg_100/00.txt', use_l2_dis=False,
              img_desc_file=None,velo_desc_file_0=None,velo_desc_file_1=None,velo_desc_file_2=None,velo_desc_file_3=None,velo_desc_file_4=None,velo_desc_file_5=None,velo_desc_file_6=None,velo_desc_file_7=None):
    net = RINet_attention()
    net.load(model_file)
    net.to(device=device)
    net.eval()
    img_desc=np.fromfile(img_desc_file, dtype=np.float32).reshape(-1,12,360)

    velo_desc_0=np.fromfile(velo_desc_file_0, dtype=np.float32).reshape(-1,12,360)
    velo_desc_1=np.fromfile(velo_desc_file_1, dtype=np.float32).reshape(-1,12,360)
    velo_desc_2=np.fromfile(velo_desc_file_2, dtype=np.float32).reshape(-1,12,360)
    velo_desc_3=np.fromfile(velo_desc_file_3, dtype=np.float32).reshape(-1,12,360)
    velo_desc_4=np.fromfile(velo_desc_file_4, dtype=np.float32).reshape(-1,12,360)
    velo_desc_5=np.fromfile(velo_desc_file_5, dtype=np.float32).reshape(-1,12,360)
    velo_desc_6=np.fromfile(velo_desc_file_6, dtype=np.float32).reshape(-1,12,360)
    velo_desc_7=np.fromfile(velo_desc_file_7, dtype=np.float32).reshape(-1,12,360)

    img_desc_o=img_desc/50.0

    velo_desc_0_o=velo_desc_0/50.0
    velo_desc_1_o=velo_desc_1/50.0
    velo_desc_2_o=velo_desc_2/50.0
    velo_desc_3_o=velo_desc_3/50.0
    velo_desc_4_o=velo_desc_4/50.0
    velo_desc_5_o=velo_desc_5/50.0
    velo_desc_6_o=velo_desc_6/50.0
    velo_desc_7_o=velo_desc_7/50.0


    img_descs_torch = torch.from_numpy(img_desc_o).to(device)
    
    velo_descs_0_torch = torch.from_numpy(velo_desc_0_o).to(device)
    velo_descs_1_torch = torch.from_numpy(velo_desc_1_o).to(device)
    velo_descs_2_torch = torch.from_numpy(velo_desc_2_o).to(device)
    velo_descs_3_torch = torch.from_numpy(velo_desc_3_o).to(device)
    velo_descs_4_torch = torch.from_numpy(velo_desc_4_o).to(device)
    velo_descs_5_torch = torch.from_numpy(velo_desc_5_o).to(device)
    velo_descs_6_torch = torch.from_numpy(velo_desc_6_o).to(device)
    velo_descs_7_torch = torch.from_numpy(velo_desc_7_o).to(device)

    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        img_descs = net.gen_feature(img_descs_torch).cpu().numpy()
        
        velo_0_descs = net.gen_feature(velo_descs_0_torch).cpu().numpy()
        velo_1_descs = net.gen_feature(velo_descs_1_torch).cpu().numpy()
        velo_2_descs = net.gen_feature(velo_descs_2_torch).cpu().numpy()
        velo_3_descs = net.gen_feature(velo_descs_3_torch).cpu().numpy()
        velo_4_descs = net.gen_feature(velo_descs_4_torch).cpu().numpy()
        velo_5_descs = net.gen_feature(velo_descs_5_torch).cpu().numpy()
        velo_6_descs = net.gen_feature(velo_descs_6_torch).cpu().numpy()
        velo_7_descs = net.gen_feature(velo_descs_7_torch).cpu().numpy()
        

        if device.type == 'cuda':
            torch.cuda.synchronize()
        
    pairs = np.genfromtxt(pair_file, dtype='int32').reshape(-1, 3)
    if use_l2_dis:

        desc1 = img_descs[pairs[:, 0]]
        desc2_list = []
        for velo_descs in [velo_0_descs, velo_1_descs, velo_2_descs, velo_3_descs,
                           velo_4_descs, velo_5_descs, velo_6_descs, velo_7_descs]:
            desc2_list.append(velo_descs[pairs[:, 1]])
        diff = max_inverse_squared_l2(desc1, desc2_list)
        label = pairs[:, 2]
        F1_max_score_split = pr_max_f1(label, diff)
    else:
        desc1 = torch.from_numpy(img_descs[pairs[:, 0]]).to(device)
        
        desc2_0 = torch.from_numpy(velo_0_descs[pairs[:, 1]]).to(device)
        desc2_1 = torch.from_numpy(velo_1_descs[pairs[:, 1]]).to(device)
        desc2_2 = torch.from_numpy(velo_2_descs[pairs[:, 1]]).to(device)
        desc2_3 = torch.from_numpy(velo_3_descs[pairs[:, 1]]).to(device)
        desc2_4 = torch.from_numpy(velo_4_descs[pairs[:, 1]]).to(device)
        desc2_5 = torch.from_numpy(velo_5_descs[pairs[:, 1]]).to(device)
        desc2_6 = torch.from_numpy(velo_6_descs[pairs[:, 1]]).to(device)
        desc2_7 = torch.from_numpy(velo_7_descs[pairs[:, 1]]).to(device)

        
        
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            scores_0, _ = net.gen_score(desc1, desc2_0)
            scores_1, _ = net.gen_score(desc1, desc2_1)
            scores_2, _ = net.gen_score(desc1, desc2_2)
            scores_3, _ = net.gen_score(desc1, desc2_3)
            scores_4, _ = net.gen_score(desc1, desc2_4)
            scores_5, _ = net.gen_score(desc1, desc2_5)
            scores_6, _ = net.gen_score(desc1, desc2_6)
            scores_7, _ = net.gen_score(desc1, desc2_7)
            
            scores_0 = scores_0.cpu().numpy()
            scores_1 = scores_1.cpu().numpy()
            scores_2 = scores_2.cpu().numpy()
            scores_3 = scores_3.cpu().numpy()
            scores_4 = scores_4.cpu().numpy()
            scores_5 = scores_5.cpu().numpy()
            scores_6 = scores_6.cpu().numpy()
            scores_7 = scores_7.cpu().numpy()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            
        gt = pairs[:, 2]
        scores_max = max_view_scores([
            scores_0, scores_1, scores_2, scores_3,
            scores_4, scores_5, scores_6, scores_7,
        ])
        F1_max_score_split = pr_max_f1(gt, scores_max)

    print("F1_split:", F1_max_score_split)
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default='00',
                        help='Sequence to eval. [default: 08]')
    parser.add_argument('--dataset', default="kitti",
                        help="Dataset (kitti or kitti360). [default: kitti]")

    parser.add_argument('--model', default=None,
                        help='Model file.')
    parser.add_argument('--velo_desc_file_0', default='/workspace/data/kitti_object/desc_kitti_split90/0/00.bin',
                        help='File of descriptors. ')
    parser.add_argument('--velo_desc_file_1', default='/workspace/data/kitti_object/desc_kitti_split90/1/00.bin',
                        help='File of descriptors. ')                    
    parser.add_argument('--velo_desc_file_2', default='/workspace/data/kitti_object/desc_kitti_split90/2/00.bin',
                        help='File of descriptors. ')     
    parser.add_argument('--velo_desc_file_3', default='/workspace/data/kitti_object/desc_kitti_split90/3/00.bin',
                        help='File of descriptors. ') 
    parser.add_argument('--velo_desc_file_4', default='/workspace/data/kitti_object/desc_kitti_split90/4/00.bin',
                        help='File of descriptors. ') 
    parser.add_argument('--velo_desc_file_5', default='/workspace/data/kitti_object/desc_kitti_split90/5/00.bin',
                        help='File of descriptors. ')         
    parser.add_argument('--velo_desc_file_6', default='/workspace/data/kitti_object/desc_kitti_split90/6/00.bin',
                        help='File of descriptors. ') 
    parser.add_argument('--velo_desc_file_7', default='/workspace/data/kitti_object/desc_kitti_split90/7/00.bin',
                        help='File of descriptors. ') 
    parser.add_argument('--img_desc_file', default='/workspace/data/kitti_object/desc_kitti_image/00.bin',
                        help='File of descriptors.')
    parser.add_argument('--pairs_file', default='/workspace/data/RINet/pairs_kitti/neg_100/00.txt',
                        help='Candidate pairs.')            
    
    parser.add_argument('--eval_type', default="f1",
                        help='Type of evaluation (f1 or recall). [default: f1]')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU id, set as CUDA_VISIBLE_DEVICES. [default: None]')
    cfg = parser.parse_args()
    if cfg.dataset == "kitti" and cfg.eval_type == "f1":
        fast_eval(seq=cfg.seq, model_file=cfg.model,
                    pair_file=cfg.pairs_file,img_desc_file=cfg.img_desc_file,
                  velo_desc_file_0=cfg.velo_desc_file_0,velo_desc_file_1=cfg.velo_desc_file_1,
                  velo_desc_file_2=cfg.velo_desc_file_2,velo_desc_file_3=cfg.velo_desc_file_3,
                  velo_desc_file_4=cfg.velo_desc_file_4,velo_desc_file_5=cfg.velo_desc_file_5,
                  velo_desc_file_6=cfg.velo_desc_file_6,velo_desc_file_7=cfg.velo_desc_file_7)
    else:
        print("Dataset and eval_type error.")
