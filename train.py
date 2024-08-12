import string
import torch
from net import RINet_attention
from database import evalDataset_kitti360, SigmoidDataset_kitti360, SigmoidDataset_train, SigmoidDataset_eval,SigmoidDataset_eval_test,SigmoidDataset_eval_eval
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import os
import argparse
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = "6"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def train(cfg):
    writer = SummaryWriter()
    net = RINet_attention()
    net.to(device=device)
    print(net)
    sequs = cfg.all_seqs
    sequs.remove(cfg.seq)
    train_dataset = SigmoidDataset_train(sequs=sequs, neg_ratio=cfg.neg_ratio,
                                         eva_ratio=cfg.eval_ratio, gt_folder=cfg.gt_folder,
                                         img_desc_folder=cfg.img_desc_folder,
                                         velo_desc_folder_0=cfg.velo_desc_folder_0,velo_desc_folder_1=cfg.velo_desc_folder_1,velo_desc_folder_2=cfg.velo_desc_folder_2,velo_desc_folder_3=cfg.velo_desc_folder_3,
                                         velo_desc_folder_4=cfg.velo_desc_folder_4,velo_desc_folder_5=cfg.velo_desc_folder_5,velo_desc_folder_6=cfg.velo_desc_folder_6,velo_desc_folder_7=cfg.velo_desc_folder_7
                                         )
    eval_dataset = SigmoidDataset_eval_eval(sequs=sequs, neg_ratio=cfg.neg_ratio*100,
                                         eva_ratio=cfg.eval_ratio, gt_folder=cfg.gt_folder,
                                         img_desc_folder=cfg.img_desc_folder,
                                         velo_desc_folder_0=cfg.velo_desc_folder_0,velo_desc_folder_1=cfg.velo_desc_folder_1,velo_desc_folder_2=cfg.velo_desc_folder_2,velo_desc_folder_3=cfg.velo_desc_folder_3,
                                         velo_desc_folder_4=cfg.velo_desc_folder_4,velo_desc_folder_5=cfg.velo_desc_folder_5,velo_desc_folder_6=cfg.velo_desc_folder_6,velo_desc_folder_7=cfg.velo_desc_folder_7)
    
    # test_dataset = SigmoidDataset_eval_test(sequs=[cfg.seq], neg_ratio=cfg.neg_ratio*100,
    #                                    eva_ratio=cfg.eval_ratio, desc_folder=cfg.desc_folder, gt_folder=cfg.gt_folder,
    #                                    velo_desc_folder=cfg.velo_desc_folder,img_desc_folder=cfg.img_desc_folder,
    #                                    velo_desc_folder_0=cfg.velo_desc_folder_0,velo_desc_folder_1=cfg.velo_desc_folder_1,velo_desc_folder_2=cfg.velo_desc_folder_2,velo_desc_folder_3=cfg.velo_desc_folder_3,
    #                                     velo_desc_folder_4=cfg.velo_desc_folder_4,velo_desc_folder_5=cfg.velo_desc_folder_5,velo_desc_folder_6=cfg.velo_desc_folder_6,velo_desc_folder_7=cfg.velo_desc_folder_7)
    batch_size = cfg.batch_size
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    eval_loader = DataLoader(
        dataset=eval_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    # test_loader = DataLoader(
    #     dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters(
    )), lr=cfg.learning_rate, weight_decay=1e-6)
    epoch = cfg.max_epoch
    starting_epoch = 0
    batch_num = 0
    if not cfg.model == "":
        checkpoint = torch.load(cfg.model)
        starting_epoch = checkpoint['epoch']
        batch_num = checkpoint['batch_num']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    for i in range(starting_epoch, epoch):
        net.train()
        pred = []
        gt = []
        for i_batch, sample_batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train epoch '+str(i), leave=False):
            optimizer.zero_grad()
            out, diff,out_cat = net(sample_batch["desc1"].to(device=device), 
                sample_batch["desc2_0"].to(device=device), 
                sample_batch["desc2_1"].to(device=device),
                sample_batch["desc2_2"].to(device=device),
                sample_batch["desc2_3"].to(device=device), 
                sample_batch["desc2_4"].to(device=device),
                sample_batch["desc2_5"].to(device=device),
                sample_batch["desc2_6"].to(device=device), 
                sample_batch["desc2_7"].to(device=device),)
            yaw_sec_gt=sample_batch["yaw_e_sec"].to(device=device)
            out_cat=out_cat.permute(1,0)
            labels = sample_batch["label"].to(device=device)
            # print('out',out)
            # print('labels',labels)

            weights=torch.zeros(out_cat.shape[1])
            for fov_i in range(len(weights)):
                #加1是为了防止当所有样本朝向都一直时，算出的权重为0
                weights[fov_i]=(1+np.sum(labels.cpu().numpy() == 1)
                                -np.sum((yaw_sec_gt.cpu().numpy() == fov_i) * (labels.cpu().numpy() == 1)))/np.sum(labels.cpu().numpy() == 1)
            # print('weights',weights)
            weights=weights.to(device=device)
            loss_ce_func = torch.nn.CrossEntropyLoss(weight=weights,reduce=False)
            loss_ce = loss_ce_func(out_cat, yaw_sec_gt.long())
            loss_ce=torch.mean(loss_ce*labels)

            loss1 = torch.nn.functional.binary_cross_entropy_with_logits(
                out, labels)
            loss2 = labels*diff*diff+(1-labels)*torch.nn.functional.relu(
                cfg.margin-diff)*torch.nn.functional.relu(cfg.margin-diff)
            loss2 = torch.mean(loss2)
            loss = loss1+loss2+loss_ce
            # loss = loss1+loss2
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                writer.add_scalar(
                    'total loss', loss.cpu().item(), global_step=batch_num)
                writer.add_scalar('loss1', loss1.cpu().item(),
                                  global_step=batch_num)
                writer.add_scalar('loss2', loss2.cpu().item(),
                                  global_step=batch_num)
                writer.add_scalar('loss_ce', loss_ce.cpu().item(),
                                  global_step=batch_num)
                batch_num += 1
                outlabel = out.cpu().numpy()
                label = sample_batch['label'].cpu().numpy()
                mask = (label > 0.9906840407) | (label < 0.0012710163)
                label = label[mask]
                label[label < 0.5] = 0
                label[label > 0.5] = 1
                pred.extend(outlabel[mask].tolist())
                gt.extend(label.tolist())
        pred = np.array(pred, dtype='float32')
        pred = np.nan_to_num(pred)
        gt = np.array(gt, dtype='float32')
        precision, recall, _ = metrics.precision_recall_curve(gt, pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        trainaccur = np.max(F1_score)
        print('Train F1:', trainaccur)
        print('i',i)
        writer.add_scalar('train f1', trainaccur, global_step=i)

        evalaccur = test(net=net, dataloader=eval_loader)
        writer.add_scalar('eval_train f1', evalaccur, global_step=i)
        print('Eval_train F1:', evalaccur)

        # lastaccur = test(net=net, dataloader=test_loader)
        # writer.add_scalar('eval_test f1', lastaccur, global_step=i)
        # print('Eval_test F1:', lastaccur)
        torch.save({'epoch': i, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(
        ), 'batch_num': batch_num}, os.path.join(cfg.log_dir, cfg.seq, str(i)+'.ckpt'))


def test(net, dataloader):
    net.eval()
    pred = []
    gt = []
    with torch.no_grad():
        for i_batch, sample_batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Eval", leave=False):
            # out, _ = net(sample_batch["desc1"].to(
            #     device=device), sample_batch["desc2"].to(device=device))
            out, _,_ =net(sample_batch["desc1"].to(device=device), 
                sample_batch["desc2_0"].to(device=device),
                sample_batch["desc2_1"].to(device=device),
                sample_batch["desc2_2"].to(device=device),
                sample_batch["desc2_3"].to(device=device),
                sample_batch["desc2_4"].to(device=device),
                sample_batch["desc2_5"].to(device=device),
                sample_batch["desc2_6"].to(device=device),
                sample_batch["desc2_7"].to(device=device),
                )
            out = out.cpu()
            outlabel = out
            label = sample_batch['label']
            mask = (label > 0.9906840407) | (label < 0.0012710163)
            label = label[mask]
            label[label < 0.5] = 0
            label[label > 0.5] = 1
            pred.extend(outlabel[mask])
            gt.extend(label)
        pred = np.array(pred, dtype='float32')
        gt = np.array(gt, dtype='float32')
        pred = np.nan_to_num(pred)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            gt, pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        testaccur = np.max(F1_score)
        return testaccur


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='log/',
                        help='Log dir. [default: log]')
    parser.add_argument('--seq', default='00',
                        help='Sequence to test. [default: 00]')
    parser.add_argument('--all_seqs', type=list, default=['00', '01', '02', '03', '04', '05', '06', '07', '08',
                        '09', '10'], help="All sequence. [default: ['00','01','02','03','04','05','06','07','08','09','10'] ]")
    parser.add_argument('--neg_ratio', type=float, default=1,
                        help='The proportion of negative samples used during training. [default: 1]')
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                        help='Proportion of samples used for validation. [default: 0.1]')

    parser.add_argument('--gt_folder', default="data/kitti/gt_split90",
                        help='Folder containing gt files. ')
    parser.add_argument('--velo_desc_folder_0', default="data/kitti/0",
                        help='Folder containing velo descriptors')
    parser.add_argument('--velo_desc_folder_1', default="data/kitti/1",
                        help='Folder containing velo descriptors')
    parser.add_argument('--velo_desc_folder_2', default="data/kitti/2",
                        help='Folder containing velo descriptors')
    parser.add_argument('--velo_desc_folder_3', default="data/kitti/3",
                        help='Folder containing velo descriptors')
    parser.add_argument('--velo_desc_folder_4', default="data/kitti/4",
                        help='Folder containing velo descriptors')
    parser.add_argument('--velo_desc_folder_5', default="data/kitti/5",
                        help='Folder containing velo descriptors')
    parser.add_argument('--velo_desc_folder_6', default="data/kitti/6",
                        help='Folder containing velo descriptors')
    parser.add_argument('--velo_desc_folder_7', default="data/kitti/7",
                        help='Folder containing velo descriptors')
    parser.add_argument('--img_desc_folder', default="data/kitti/desc_image",
                        help='Folder containing img descriptors')

    parser.add_argument('--model', default="",
                        help='Pretrained model. [default: ""]')
    parser.add_argument('--max_epoch', type=int, default=20,
                        help='Epoch to run. [default: 20]')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch Size during training. [default: 1024]')
    parser.add_argument('--learning_rate', type=float, default=0.02,
                        help='Initial learning rate. [default: 0.02]')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-6, help='Weight decay. [default: 1e-6]')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Margin used in contrastive loss. [default: 0.2]')
    cfg = parser.parse_args()
    if(not os.path.exists(os.path.join(cfg.log_dir, cfg.seq))):
        os.makedirs(os.path.join(cfg.log_dir, cfg.seq))
    train(cfg)
