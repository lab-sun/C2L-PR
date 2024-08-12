import torch
from net import RINet_attention
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn import metrics
from matplotlib import pyplot as plt
import sys
import time
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "6"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def recall(seq='00', model_file="./model/attention/00.ckpt",
             img_desc_file=None,
            velo_desc_file_0=None,velo_desc_file_1=None,velo_desc_file_2=None,
            velo_desc_file_3=None,velo_desc_file_4=None,velo_desc_file_5=None,
            velo_desc_file_6=None,velo_desc_file_7=None,
            pose_file="./data/pose_kitti/00.txt"):
    poses = np.genfromtxt(pose_file)
    poses = poses[:, [3, 11]]
    inner = 2*np.matmul(poses, poses.T)
    xx = np.sum(poses**2, 1, keepdims=True)
    dis = xx-inner+xx.T
    dis = np.sqrt(np.abs(dis))
    id_pos = np.argwhere(dis <= 5)
    id_pos = id_pos[id_pos[:, 0]-id_pos[:, 1] > 50]
    pos_dict = {}
    for v in id_pos:
        if v[0] in pos_dict.keys():
            pos_dict[v[0]].append(v[1])
        else:
            pos_dict[v[0]] = [v[1]]
    
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

    net = RINet_attention()
    net.load(model_file)
    net.to(device=device)
    net.eval()
    out_save = []
    recall = np.array([0.]*25)
    for v in tqdm(pos_dict.keys()):
        print('v',v)
        candidates_0 = []
        candidates_1 = []
        candidates_2 = []
        candidates_3 = []
        candidates_4 = []
        candidates_5 = []
        candidates_6 = []
        candidates_7 = []
        targets = []
        for c in range(0, v-50):
            candidates_0.append(velo_desc_0_o[c])
            candidates_1.append(velo_desc_1_o[c])
            candidates_2.append(velo_desc_2_o[c])
            candidates_3.append(velo_desc_3_o[c])
            candidates_4.append(velo_desc_4_o[c])
            candidates_5.append(velo_desc_5_o[c])
            candidates_6.append(velo_desc_6_o[c])
            candidates_7.append(velo_desc_7_o[c])
            targets.append(img_desc_o[v])
        candidates_0 = np.array(candidates_0, dtype='float32')
        candidates_1 = np.array(candidates_1, dtype='float32')
        candidates_2 = np.array(candidates_2, dtype='float32')
        candidates_3 = np.array(candidates_3, dtype='float32')
        candidates_4 = np.array(candidates_4, dtype='float32')
        candidates_5 = np.array(candidates_5, dtype='float32')
        candidates_6 = np.array(candidates_6, dtype='float32')
        candidates_7 = np.array(candidates_7, dtype='float32')
        targets = np.array(targets, dtype='float32')
        candidates_0 = torch.from_numpy(candidates_0)
        candidates_1 = torch.from_numpy(candidates_1)
        candidates_2 = torch.from_numpy(candidates_2)
        candidates_3 = torch.from_numpy(candidates_3)
        candidates_4 = torch.from_numpy(candidates_4)
        candidates_5 = torch.from_numpy(candidates_5)
        candidates_6 = torch.from_numpy(candidates_6)
        candidates_7 = torch.from_numpy(candidates_7)
        targets = torch.from_numpy(targets)
        with torch.no_grad():
            out, _,_ = net(targets.to(device=device),
                         candidates_0.to(device=device),
                         candidates_1.to(device=device),
                         candidates_2.to(device=device),
                         candidates_3.to(device=device),
                         candidates_4.to(device=device),
                         candidates_5.to(device=device),
                         candidates_6.to(device=device),
                         candidates_7.to(device=device),
                         )
            out = out.cpu().numpy()
            ids = np.argsort(-out)
            o = [v]
            o += ids[:25].tolist()
            out_save.append(o)
            for i in range(25):
                if ids[i] in pos_dict[v]:
                    recall[i:] += 1
                    break
    if not os.path.exists('result'):
        os.mkdir('result')
    np.savetxt(os.path.join('result', seq+'_recall.txt'), out_save, fmt='%d')
    recall /= len(pos_dict.keys())
    print(recall)
    np.savetxt(os.path.join('result', seq+'_recall_scores.txt'), recall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default='00',
                        help='Sequence to eval. [default: 08]')
    parser.add_argument('--dataset', default="kitti",
                        help="Dataset (kitti or kitti360). [default: kitti]")
                        
    parser.add_argument('--model', default=None,
                        help='Model file. ')
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
    parser.add_argument('--pose_file', default="/workspace/data/RINet/pose_kitti/00.txt",
                        help='Pose file (eval_type=recall). ')

    parser.add_argument('--eval_type', default="recall",
                        help='Type of evaluation (f1 or recall). [default: f1]')

    cfg = parser.parse_args()
    if cfg.dataset == "kitti" and cfg.eval_type == "recall":
        recall(seq=cfg.seq, model_file=cfg.model, 
                img_desc_file=cfg.img_desc_file,
                velo_desc_file_0=cfg.velo_desc_file_0,velo_desc_file_1=cfg.velo_desc_file_1,
                velo_desc_file_2=cfg.velo_desc_file_2,velo_desc_file_3=cfg.velo_desc_file_3,
                velo_desc_file_4=cfg.velo_desc_file_4,velo_desc_file_5=cfg.velo_desc_file_5,
                velo_desc_file_6=cfg.velo_desc_file_6,velo_desc_file_7=cfg.velo_desc_file_7,
                pose_file=cfg.pose_file)
    else:
        print("Dataset and eval_type error.")
