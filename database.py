from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import json
import random

def load_bin(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    # print('scan',scan)
    # print('scan.shape',scan.shape)
    # os._exit()
    # scan = scan.reshape((4541,12,360))
    return scan

class SigmoidDataset_eval(Dataset):
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], neg_ratio=1, desc_folder="./data/desc_kitti", gt_folder="./data/gt_kitti", eva_ratio=0.1,
                velo_desc_folder=None,img_desc_folder=None) -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0

        self.velo_descs = []
        self.img_descs = []

        for seq in sequs:
            # print('eval:seq',seq)
            desc_file = os.path.join(desc_folder, seq+'.npy')
            gt_file = os.path.join(gt_folder, seq+'.npz')

            velo_desc_file=os.path.join(velo_desc_folder, seq+'.bin')
            img_desc_file=os.path.join(img_desc_folder, seq+'.bin')
            velo_desc=np.fromfile(velo_desc_file, dtype=np.float32).reshape(-1,12,360)
            img_desc=np.fromfile(img_desc_file, dtype=np.float32).reshape(-1,12,360)
            self.velo_descs.append(velo_desc)
            self.img_descs.append(img_desc)

            # self.descs.append(np.load(desc_file))         
            gt = np.load(gt_file)
            # print('gt[pos]',len(gt['pos']))
            pos = gt['pos'][-int(len(gt['pos'])*eva_ratio):]
            # print('pos',len(pos))
            neg = gt['neg'][-int(len(gt['neg'])*eva_ratio):]
            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            # print('pos_num',len(self.gt_pos[-1]))
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

        # print('eval:self.pos_num+self.neg_num',self.pos_num+self.neg_num)
        # os._exit()


    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            # out = {"desc1": self.descs[int(id_seq)][int(
            #     pair[0])]/50., "desc2": self.descs[int(id_seq)][int(pair[1])]/50., 'label': pair[2]}
            out = {"desc1": self.img_descs[int(id_seq)][int(
                pair[0])]/50., "desc2": self.velo_descs[int(id_seq)][int(pair[1])]/50., 'label': pair[2]}
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                # out = {"desc1": self.descs[i-1][int(
                #     pair[0])]/50., "desc2": self.descs[i-1][int(pair[1])]/50., 'label': pair[2]}
                out = {"desc1": self.img_descs[i-1][int(
                    pair[0])]/50., "desc2": self.velo_descs[i-1][int(pair[1])]/50., 'label': pair[2]}
                return out

class SigmoidDataset_eval_test(Dataset):
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], neg_ratio=1, desc_folder="./data/desc_kitti", gt_folder="./data/gt_kitti", eva_ratio=0.1,
                velo_desc_folder=None,img_desc_folder=None,
                velo_desc_folder_0=None,velo_desc_folder_1=None,velo_desc_folder_2=None,velo_desc_folder_3=None,
                velo_desc_folder_4=None,velo_desc_folder_5=None,velo_desc_folder_6=None,velo_desc_folder_7=None,) -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0

        self.velo_descs = []
        self.img_descs = []
        
        self.velo_descs_0 = []
        self.velo_descs_1 = []
        self.velo_descs_2 = []
        self.velo_descs_3 = []
        self.velo_descs_4 = []
        self.velo_descs_5 = []
        self.velo_descs_6 = []
        self.velo_descs_7 = []

        for seq in sequs:
            desc_file = os.path.join(desc_folder, seq+'.npy')
            gt_file = os.path.join(gt_folder, seq+'.npz')

            velo_desc_file_0=os.path.join(velo_desc_folder_0, seq+'.bin')
            velo_desc_file_1=os.path.join(velo_desc_folder_1, seq+'.bin')
            velo_desc_file_2=os.path.join(velo_desc_folder_2, seq+'.bin')
            velo_desc_file_3=os.path.join(velo_desc_folder_3, seq+'.bin')
            velo_desc_file_4=os.path.join(velo_desc_folder_4, seq+'.bin')
            velo_desc_file_5=os.path.join(velo_desc_folder_5, seq+'.bin')
            velo_desc_file_6=os.path.join(velo_desc_folder_6, seq+'.bin')
            velo_desc_file_7=os.path.join(velo_desc_folder_7, seq+'.bin')


            velo_desc_file=os.path.join(velo_desc_folder, seq+'.bin')
            img_desc_file=os.path.join(img_desc_folder, seq+'.bin')
            velo_desc=np.fromfile(velo_desc_file, dtype=np.float32).reshape(-1,12,360)
            img_desc=np.fromfile(img_desc_file, dtype=np.float32).reshape(-1,12,360)
            self.velo_descs.append(velo_desc)
            self.img_descs.append(img_desc)


            velo_desc_0=np.fromfile(velo_desc_file_0, dtype=np.float32).reshape(-1,12,360)
            velo_desc_1=np.fromfile(velo_desc_file_1, dtype=np.float32).reshape(-1,12,360)
            velo_desc_2=np.fromfile(velo_desc_file_2, dtype=np.float32).reshape(-1,12,360)
            velo_desc_3=np.fromfile(velo_desc_file_3, dtype=np.float32).reshape(-1,12,360)
            velo_desc_4=np.fromfile(velo_desc_file_4, dtype=np.float32).reshape(-1,12,360)
            velo_desc_5=np.fromfile(velo_desc_file_5, dtype=np.float32).reshape(-1,12,360)
            velo_desc_6=np.fromfile(velo_desc_file_6, dtype=np.float32).reshape(-1,12,360)
            velo_desc_7=np.fromfile(velo_desc_file_7, dtype=np.float32).reshape(-1,12,360)


            self.velo_descs_0.append(velo_desc_0)
            self.velo_descs_1.append(velo_desc_1)
            self.velo_descs_2.append(velo_desc_2)
            self.velo_descs_3.append(velo_desc_3)
            self.velo_descs_4.append(velo_desc_4)
            self.velo_descs_5.append(velo_desc_5)
            self.velo_descs_6.append(velo_desc_6)
            self.velo_descs_7.append(velo_desc_7)

            # self.descs.append(np.load(desc_file))         
            gt = np.load(gt_file)
            # print('gt[pos]',len(gt['pos']))
            # os._exit()
            pos = gt['pos'][:-int(len(gt['pos'])*eva_ratio)]
            # print('pos',len(pos))
            neg = gt['neg'][:-int(len(gt['neg'])*eva_ratio)]
            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            # print('pos_num',len(self.gt_pos[-1]))
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

        # print('eval:self.pos_num+self.neg_num',self.pos_num+self.neg_num)
        # os._exit()


    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            # out = {"desc1": self.descs[int(id_seq)][int(
            #     pair[0])]/50., "desc2": self.descs[int(id_seq)][int(pair[1])]/50., 'label': pair[2]}
            # out = {"desc1": self.img_descs[int(id_seq)][int(
            #     pair[0])]/50., "desc2": self.velo_descs[int(id_seq)][int(pair[1])]/50., 'label': pair[2]}
            out = {"desc1": self.img_descs[int(id_seq)][int(pair[0])]/50., 
                    "desc2": self.velo_descs[int(id_seq)][int(pair[1])]/50., 
                    "desc2_0": self.velo_descs_0[int(id_seq)][int(pair[1])]/50., 
                    "desc2_1": self.velo_descs_1[int(id_seq)][int(pair[1])]/50., 
                    "desc2_2": self.velo_descs_2[int(id_seq)][int(pair[1])]/50., 
                    "desc2_3": self.velo_descs_3[int(id_seq)][int(pair[1])]/50.,
                    "desc2_4": self.velo_descs_4[int(id_seq)][int(pair[1])]/50.,
                    "desc2_5": self.velo_descs_5[int(id_seq)][int(pair[1])]/50.,
                    "desc2_6": self.velo_descs_6[int(id_seq)][int(pair[1])]/50.,
                    "desc2_7": self.velo_descs_7[int(id_seq)][int(pair[1])]/50., 
                    'label': pair[2]}
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                # out = {"desc1": self.descs[i-1][int(
                #     pair[0])]/50., "desc2": self.descs[i-1][int(pair[1])]/50., 'label': pair[2]}
                # out = {"desc1": self.img_descs[i-1][int(
                #     pair[0])]/50., "desc2": self.velo_descs[i-1][int(pair[1])]/50., 'label': pair[2]}
                out = {"desc1": self.img_descs[i-1][int(pair[0])]/50., 
                        "desc2": self.velo_descs[i-1][int(pair[1])]/50., 
                        "desc2_0": self.velo_descs_0[i-1][int(pair[1])]/50., 
                        "desc2_1": self.velo_descs_1[i-1][int(pair[1])]/50., 
                        "desc2_2": self.velo_descs_2[i-1][int(pair[1])]/50., 
                        "desc2_3": self.velo_descs_3[i-1][int(pair[1])]/50.,
                        "desc2_4": self.velo_descs_4[i-1][int(pair[1])]/50.,
                        "desc2_5": self.velo_descs_5[i-1][int(pair[1])]/50., 
                        "desc2_6": self.velo_descs_6[i-1][int(pair[1])]/50.,
                        "desc2_7": self.velo_descs_7[i-1][int(pair[1])]/50., 
                        'label': pair[2]}
                return out

class SigmoidDataset_eval_eval(Dataset): 
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], neg_ratio=1, gt_folder=None, eva_ratio=0.1,
                velo_desc_folder=None,img_desc_folder=None,
                velo_desc_folder_0=None,velo_desc_folder_1=None,velo_desc_folder_2=None,velo_desc_folder_3=None,
                velo_desc_folder_4=None,velo_desc_folder_5=None,velo_desc_folder_6=None,velo_desc_folder_7=None,
                ) -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0

        self.velo_descs = []
        self.img_descs = []

        self.velo_descs_0 = []
        self.velo_descs_1 = []
        self.velo_descs_2 = []
        self.velo_descs_3 = []
        self.velo_descs_4 = []
        self.velo_descs_5 = []
        self.velo_descs_6 = []
        self.velo_descs_7 = []

        
        for seq in sequs:
            print('train:seq',seq)
            gt_file = os.path.join(gt_folder, seq+'.npz')
            img_desc_file=os.path.join(img_desc_folder, seq+'.bin')


            velo_desc_file_0=os.path.join(velo_desc_folder_0, seq+'.bin')
            velo_desc_file_1=os.path.join(velo_desc_folder_1, seq+'.bin')
            velo_desc_file_2=os.path.join(velo_desc_folder_2, seq+'.bin')
            velo_desc_file_3=os.path.join(velo_desc_folder_3, seq+'.bin')
            velo_desc_file_4=os.path.join(velo_desc_folder_4, seq+'.bin')
            velo_desc_file_5=os.path.join(velo_desc_folder_5, seq+'.bin')
            velo_desc_file_6=os.path.join(velo_desc_folder_6, seq+'.bin')
            velo_desc_file_7=os.path.join(velo_desc_folder_7, seq+'.bin')
            
            # velo_desc=np.fromfile(velo_desc_file, dtype=np.float32).reshape(-1,12,360)
            img_desc=np.fromfile(img_desc_file, dtype=np.float32).reshape(-1,12,360)

            velo_desc_0=np.fromfile(velo_desc_file_0, dtype=np.float32).reshape(-1,12,360)
            velo_desc_1=np.fromfile(velo_desc_file_1, dtype=np.float32).reshape(-1,12,360)
            velo_desc_2=np.fromfile(velo_desc_file_2, dtype=np.float32).reshape(-1,12,360)
            velo_desc_3=np.fromfile(velo_desc_file_3, dtype=np.float32).reshape(-1,12,360)
            velo_desc_4=np.fromfile(velo_desc_file_4, dtype=np.float32).reshape(-1,12,360)
            velo_desc_5=np.fromfile(velo_desc_file_5, dtype=np.float32).reshape(-1,12,360)
            velo_desc_6=np.fromfile(velo_desc_file_6, dtype=np.float32).reshape(-1,12,360)
            velo_desc_7=np.fromfile(velo_desc_file_7, dtype=np.float32).reshape(-1,12,360)
            
            # self.velo_descs.append(velo_desc)
            self.img_descs.append(img_desc)

            self.velo_descs_0.append(velo_desc_0)
            self.velo_descs_1.append(velo_desc_1)
            self.velo_descs_2.append(velo_desc_2)
            self.velo_descs_3.append(velo_desc_3)
            self.velo_descs_4.append(velo_desc_4)
            self.velo_descs_5.append(velo_desc_5)
            self.velo_descs_6.append(velo_desc_6)
            self.velo_descs_7.append(velo_desc_7)

            gt = np.load(gt_file)

            pos = gt['pos'][-int(len(gt['pos'])*eva_ratio):]
            neg = gt['neg'][-int(len(gt['neg'])*eva_ratio):]


            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)


    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {"desc1": self.img_descs[int(id_seq)][int(pair[0])]/50., 
                "desc2_0": self.velo_descs_0[int(id_seq)][int(pair[1])]/50.,
                "desc2_1": self.velo_descs_1[int(id_seq)][int(pair[1])]/50.,
                "desc2_2": self.velo_descs_2[int(id_seq)][int(pair[1])]/50.,
                "desc2_3": self.velo_descs_3[int(id_seq)][int(pair[1])]/50.,
                "desc2_4": self.velo_descs_4[int(id_seq)][int(pair[1])]/50.,
                "desc2_5": self.velo_descs_5[int(id_seq)][int(pair[1])]/50.,
                "desc2_6": self.velo_descs_6[int(id_seq)][int(pair[1])]/50.,
                "desc2_7": self.velo_descs_7[int(id_seq)][int(pair[1])]/50.,
                'label': pair[2],'yaw_e_sec':pair[4]}
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {"desc1": self.img_descs[i-1][int(pair[0])]/50.,
                "desc2_0": self.velo_descs_0[i-1][int(pair[1])]/50., 
                "desc2_1": self.velo_descs_1[i-1][int(pair[1])]/50., 
                "desc2_2": self.velo_descs_2[i-1][int(pair[1])]/50., 
                "desc2_3": self.velo_descs_3[i-1][int(pair[1])]/50., 
                "desc2_4": self.velo_descs_4[i-1][int(pair[1])]/50., 
                "desc2_5": self.velo_descs_5[i-1][int(pair[1])]/50., 
                "desc2_6": self.velo_descs_6[i-1][int(pair[1])]/50., 
                "desc2_7": self.velo_descs_7[i-1][int(pair[1])]/50.,
                'label': pair[2],'yaw_e_sec':pair[4]}
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0


class SigmoidDataset_train(Dataset): 
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], neg_ratio=1, gt_folder="./data/gt_kitti", eva_ratio=0.1,
                velo_desc_folder=None,img_desc_folder=None,
                velo_desc_folder_0=None,velo_desc_folder_1=None,velo_desc_folder_2=None,velo_desc_folder_3=None,
                velo_desc_folder_4=None,velo_desc_folder_5=None,velo_desc_folder_6=None,velo_desc_folder_7=None,
                ) -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0

        self.velo_descs = []
        self.img_descs = []

        self.velo_descs_0 = []
        self.velo_descs_1 = []
        self.velo_descs_2 = []
        self.velo_descs_3 = []
        self.velo_descs_4 = []
        self.velo_descs_5 = []
        self.velo_descs_6 = []
        self.velo_descs_7 = []

        
        for seq in sequs:
            print('train:seq',seq)
            gt_file = os.path.join(gt_folder, seq+'.npz')
            img_desc_file=os.path.join(img_desc_folder, seq+'.bin')

            velo_desc_file_0=os.path.join(velo_desc_folder_0, seq+'.bin')
            velo_desc_file_1=os.path.join(velo_desc_folder_1, seq+'.bin')
            velo_desc_file_2=os.path.join(velo_desc_folder_2, seq+'.bin')
            velo_desc_file_3=os.path.join(velo_desc_folder_3, seq+'.bin')
            velo_desc_file_4=os.path.join(velo_desc_folder_4, seq+'.bin')
            velo_desc_file_5=os.path.join(velo_desc_folder_5, seq+'.bin')
            velo_desc_file_6=os.path.join(velo_desc_folder_6, seq+'.bin')
            velo_desc_file_7=os.path.join(velo_desc_folder_7, seq+'.bin')
          
            img_desc=np.fromfile(img_desc_file, dtype=np.float32).reshape(-1,12,360)

            velo_desc_0=np.fromfile(velo_desc_file_0, dtype=np.float32).reshape(-1,12,360)
            velo_desc_1=np.fromfile(velo_desc_file_1, dtype=np.float32).reshape(-1,12,360)
            velo_desc_2=np.fromfile(velo_desc_file_2, dtype=np.float32).reshape(-1,12,360)
            velo_desc_3=np.fromfile(velo_desc_file_3, dtype=np.float32).reshape(-1,12,360)
            velo_desc_4=np.fromfile(velo_desc_file_4, dtype=np.float32).reshape(-1,12,360)
            velo_desc_5=np.fromfile(velo_desc_file_5, dtype=np.float32).reshape(-1,12,360)
            velo_desc_6=np.fromfile(velo_desc_file_6, dtype=np.float32).reshape(-1,12,360)
            velo_desc_7=np.fromfile(velo_desc_file_7, dtype=np.float32).reshape(-1,12,360)

            self.img_descs.append(img_desc)

            self.velo_descs_0.append(velo_desc_0)
            self.velo_descs_1.append(velo_desc_1)
            self.velo_descs_2.append(velo_desc_2)
            self.velo_descs_3.append(velo_desc_3)
            self.velo_descs_4.append(velo_desc_4)
            self.velo_descs_5.append(velo_desc_5)
            self.velo_descs_6.append(velo_desc_6)
            self.velo_descs_7.append(velo_desc_7)

            gt = np.load(gt_file)
            pos = gt['pos'][:-int(len(gt['pos'])*eva_ratio)]
            neg = gt['neg'][:-int(len(gt['neg'])*eva_ratio)]
            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)


    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {"desc1": self.img_descs[int(id_seq)][int(pair[0])]/50., 
                "desc2_0": self.velo_descs_0[int(id_seq)][int(pair[1])]/50.,
                "desc2_1": self.velo_descs_1[int(id_seq)][int(pair[1])]/50.,
                "desc2_2": self.velo_descs_2[int(id_seq)][int(pair[1])]/50.,
                "desc2_3": self.velo_descs_3[int(id_seq)][int(pair[1])]/50.,
                "desc2_4": self.velo_descs_4[int(id_seq)][int(pair[1])]/50.,
                "desc2_5": self.velo_descs_5[int(id_seq)][int(pair[1])]/50.,
                "desc2_6": self.velo_descs_6[int(id_seq)][int(pair[1])]/50.,
                "desc2_7": self.velo_descs_7[int(id_seq)][int(pair[1])]/50.,
                'label': pair[2],'yaw_e_sec':pair[4]}
            if random.randint(0, 1) > 0:
                self.rand_occ(out["desc1"])
                self.rand_occ(out["desc2_0"])
                self.rand_occ(out["desc2_1"])
                self.rand_occ(out["desc2_2"])
                self.rand_occ(out["desc2_3"])
                self.rand_occ(out["desc2_4"])
                self.rand_occ(out["desc2_5"])
                self.rand_occ(out["desc2_6"])
                self.rand_occ(out["desc2_7"])
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {"desc1": self.img_descs[i-1][int(pair[0])]/50., 
                "desc2_0": self.velo_descs_0[i-1][int(pair[1])]/50., 
                "desc2_1": self.velo_descs_1[i-1][int(pair[1])]/50., 
                "desc2_2": self.velo_descs_2[i-1][int(pair[1])]/50., 
                "desc2_3": self.velo_descs_3[i-1][int(pair[1])]/50., 
                "desc2_4": self.velo_descs_4[i-1][int(pair[1])]/50., 
                "desc2_5": self.velo_descs_5[i-1][int(pair[1])]/50., 
                "desc2_6": self.velo_descs_6[i-1][int(pair[1])]/50., 
                "desc2_7": self.velo_descs_7[i-1][int(pair[1])]/50.,
                'label': pair[2],'yaw_e_sec':pair[4]}
                if random.randint(0, 1) > 0:
                    self.rand_occ(out["desc1"])
                    self.rand_occ(out["desc2_0"])
                    self.rand_occ(out["desc2_1"])
                    self.rand_occ(out["desc2_2"])
                    self.rand_occ(out["desc2_3"])
                    self.rand_occ(out["desc2_4"])
                    self.rand_occ(out["desc2_5"])
                    self.rand_occ(out["desc2_6"])
                    self.rand_occ(out["desc2_7"])
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0


class SigmoidDataset(Dataset):
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], neg_ratio=1, desc_folder="./data/desc_kitti", gt_folder="./data/gt_kitti") -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0
        for seq in sequs:
            desc_file = os.path.join(desc_folder, seq+'.npy')
            gt_file = os.path.join(gt_folder, seq+'.npz')
            self.descs.append(np.load(desc_file))
            gt = np.load(gt_file)
            self.gt_pos.append(gt['pos'])
            self.gt_neg.append(gt['neg'])
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {"desc1": self.descs[int(id_seq)][int(
                pair[0])]/50., "desc2": self.descs[int(id_seq)][int(pair[1])]/50., 'label': pair[2]*1.}
            if random.randint(0, 2) > 1:
                self.rand_occ(out["desc1"])
                self.rand_occ(out["desc2"])
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {"desc1": self.descs[i-1][int(pair[0])]/50., "desc2": self.descs[i-1][int(
                    pair[1])]/50., 'label': pair[2]*1.}
                if random.randint(0, 2) > 1:
                    self.rand_occ(out["desc1"])
                    self.rand_occ(out["desc2"])
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0


class evalDataset(Dataset):
    def __init__(self, seq="00", desc_folder="./data/desc_kitti", gt_folder="./data/pairs_kitti/neg_100") -> None:
        super().__init__()
        self.descs = []
        self.pairs = []
        self.num = 0
        desc_file = os.path.join(desc_folder, seq+'.npy')
        pair_file = os.path.join(gt_folder, seq+'.txt')
        self.descs = np.load(desc_file)
        self.pairs = np.genfromtxt(pair_file, dtype='int32')
        self.num = len(self.pairs)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = self.pairs[idx]
        out = {"desc1": self.descs[int(
            pair[0])]/50., "desc2": self.descs[int(pair[1])]/50., 'label': pair[2]}
        angle1 = np.random.randint(0, 359)
        angle2 = np.random.randint(0, 359)
        out["desc1"] = np.roll(out["desc1"], angle1, axis=1)
        out["desc2"] = np.roll(out["desc2"], angle2, axis=1)
        return out


class SigmoidDataset_kitti360(Dataset):
    def __init__(self, sequs=['0000', '0002', '0003', '0004', '0005', '0006', '0007', '0009', '0010'], neg_ratio=1, desc_folder="./data/desc_kitti360", gt_folder="./data/gt_kitti360") -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.key_map = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0
        for seq in sequs:
            desc_file = os.path.join(desc_folder, seq+'.npy')
            gt_file = os.path.join(gt_folder, seq+'.npz')
            self.descs.append(np.load(desc_file))
            self.key_map.append(
                json.load(open(os.path.join(desc_folder, seq+'.json'))))
            gt = np.load(gt_file)
            self.gt_pos.append(gt['pos'])
            self.gt_neg.append(gt['neg'])
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {"desc1": self.descs[int(id_seq)][self.key_map[int(id_seq)][str(int(
                pair[0]))]]/50., "desc2": self.descs[int(id_seq)][self.key_map[int(id_seq)][str(int(pair[1]))]]/50., 'label': pair[2]}
            if random.randint(0, 1) > 0:
                self.rand_occ(out["desc1"])
                self.rand_occ(out["desc2"])
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {"desc1": self.descs[i-1][self.key_map[i-1][str(int(
                    pair[0]))]]/50., "desc2": self.descs[i-1][self.key_map[i-1][str(int(pair[1]))]]/50., 'label': pair[2]}
                if random.randint(0, 1) > 0:
                    self.rand_occ(out["desc1"])
                    self.rand_occ(out["desc2"])
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0


class evalDataset_kitti360(Dataset):
    def __init__(self, seq="0000", desc_folder="./data/desc_kitti360", gt_folder="./data/pairs_kitti360/neg10") -> None:
        super().__init__()
        self.descs = []
        self.pairs = []
        self.num = 0
        desc_file = os.path.join(desc_folder, seq+'.npy')
        pair_file = os.path.join(gt_folder, seq+'.txt')
        self.descs = np.load(desc_file)
        self.pairs = np.genfromtxt(pair_file, dtype='int32')
        self.key_map = json.load(open(os.path.join(desc_folder, seq+'.json')))
        self.num = len(self.pairs)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = self.pairs[idx]
        out = {"desc1": self.descs[self.key_map[str(int(
            pair[0]))]]/50., "desc2": self.descs[self.key_map[str(int(pair[1]))]]/50., 'label': pair[2]}
        return out


if __name__ == '__main__':
    database = SigmoidDataset_train(
        ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 2)
    print(len(database))
    for i in range(0, len(database)):
        idx = random.randint(0, len(database)-1)
        d = database[idx]
        print(i, d['label'])
        plt.subplot(2, 1, 1)
        plt.imshow(d['desc1'])
        plt.subplot(2, 1, 2)
        plt.imshow(d['desc2'])
        plt.show()
