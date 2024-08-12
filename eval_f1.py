import torch
from net import RINet_attention
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn import metrics
# import matplotlib
# matplotlib.use('TkAgg')
import sys
import time
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "6"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        

        torch.cuda.synchronize()
        
    pairs = np.genfromtxt(pair_file, dtype='int32').reshape(-1, 3)
    if use_l2_dis:

        desc1 = img_descs[pairs[:, 0]]
        desc2 = velo_descs[pairs[:, 1]]

        diff = desc1-desc2
        diff = 1./np.sum(diff*diff, axis=1)
        diff = diff.reshape(-1, 1)
        diff = np.nan_to_num(diff)
        label = pairs[:, 2].reshape(-1, 1)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            label, diff)
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

            torch.cuda.synchronize()
            
        gt = pairs[:, 2].reshape(-1, 1)
        
        scores_concat=np.concatenate([scores_0.reshape(-1, 1), scores_1.reshape(-1, 1),scores_2.reshape(-1, 1),scores_3.reshape(-1, 1),scores_4.reshape(-1, 1),scores_5.reshape(-1, 1),scores_6.reshape(-1, 1),scores_7.reshape(-1, 1)], axis=1)
        scores_max=np.max(scores_concat, axis=1)
        
        precision_split, recall_split, pr_thresholds_split = metrics.precision_recall_curve(
            gt, scores_max)


    F1_score_split = 2 * precision_split * recall_split / (precision_split + recall_split)
    F1_score_split = np.nan_to_num(F1_score_split)
    F1_max_score_split = np.max(F1_score_split)
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
