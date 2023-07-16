
 
#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import random
import numpy as np
import sys

bceloss = torch.nn.BCELoss(reduction='none')
kl_loss = torch.nn.KLDivLoss()
seed = 3377
savefig = False
get_hd = True
model_save = False
if True:
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False

import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import networks.deeplabv3 as netd
import networks.deeplabv3_eval as netd_eval
from tool import pyutils

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='../logs/source/checkpoint_185.pth.tar')#D4/checkpoint_170.pth.tar
    parser.add_argument('--dataset', type=str, default='Domain2')#Domain1
    parser.add_argument('--source', type=str, default='Domain1')#Domain4
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='../../../Data/Fundus/')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    parser.add_argument('--pseudo',type=str,default='../generate_pseudo/pseudolabel_D2.npz')
    parser.add_argument('--radius',type=int,default=4)

    args = parser.parse_args()
    radius = args.radius
    num_epochs = 16
    tao=0.05
    gamma=2

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_train = transforms.Compose([
        tr.Resize1(512, 32),
        tr.add_salt_pepper_noise_sim(),
        tr.adjust_light_sim(),
        tr.eraser_sim(),
        tr.Normalize_tf1(),
        tr.ToTensor1()
    ])
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_train = DL.FundusSegmentation_wsim(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_train, pseudo=args.pseudo,radius=radius)
    #db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=composed_transforms_test)

    train_loader = DataLoader(db_train, batch_size=8, shuffle=False, num_workers=0)
    #test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    # 2. model
    model = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn, radius=radius)
    
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.train()


    optim_gen = torch.optim.Adam(model.get_scratch_parameters(), lr=3e-2, betas=(0.9, 0.99))#0.002
    #optim_gen.load_state_dict(checkpoint['optim_state_dict'])
    
    best_val_cup_dice = 0.0;
    best_val_disc_dice = 0.0;
    best_avg = 0.0
    iter_num = 0
    avg_meter_cup = pyutils.AverageMeter('loss', 'bg_loss', 'fg_loss', 'neg_loss', 'bg_cnt', 'fg_cnt', 'neg_cnt')
    avg_meter_disc = pyutils.AverageMeter('loss', 'bg_loss', 'fg_loss', 'neg_loss', 'bg_cnt', 'fg_cnt', 'neg_cnt')
    for epoch_num in tqdm.tqdm(range(num_epochs), ncols=70):
        model.train()
        for batch_idx, (sample) in enumerate(train_loader):
            data, label_cup, label_disc, img_name, gt_cup, gt_disc = sample

            if torch.cuda.is_available():
                data, label_cup, label_disc = data.cuda(), label_cup, label_disc
                #data, label_cup, label_disc = data.cuda(), gt_cup, gt_disc
            #data, target = Variable(data), Variable(target)

            prediction, _, feature, aff_cup, aff_disc = model(data)
            
            optim_gen.zero_grad()

            bg_label = label_cup[0].cuda(non_blocking=True)
            fg_label = label_cup[1].cuda(non_blocking=True)
            neg_label = label_cup[2].cuda(non_blocking=True)

            bg_count = torch.sum(bg_label) + 1e-5
            fg_count = torch.sum(fg_label) + 1e-5
            neg_count = torch.sum(neg_label) + 1e-5

            bg_loss = torch.sum(- bg_label * torch.log(aff_cup + 1e-5)) / bg_count
            fg_loss = torch.sum(- fg_label * torch.log(aff_cup + 1e-5)) / fg_count
            neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff_cup)) / neg_count

            loss_cup = bg_loss/4 + fg_loss/4 + neg_loss/2
            #disc
            bg_label = label_disc[0].cuda(non_blocking=True)
            fg_label = label_disc[1].cuda(non_blocking=True)
            neg_label = label_disc[2].cuda(non_blocking=True)

            bg_count = torch.sum(bg_label) + 1e-5
            fg_count = torch.sum(fg_label) + 1e-5
            neg_count = torch.sum(neg_label) + 1e-5

            bg_loss = torch.sum(- bg_label * torch.log(aff_disc + 1e-5)) / bg_count
            fg_loss = torch.sum(- fg_label * torch.log(aff_disc + 1e-5)) / fg_count
            neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff_disc)) / neg_count

            loss_disc = bg_loss/4 + fg_loss/4 + neg_loss/2

            loss_aff = (loss_cup + loss_disc)/2
            

            loss = loss_aff 
            loss.backward()
            optim_gen.step()
            iter_num = iter_num + 1
            
        #test
        
        model_eval = model
        model_eval.eval()
        
        with torch.no_grad():
            for batch_idx, (sample) in enumerate(train_loader):
                data, label_cup, label_disc, img_name, gt_cup, gt_disc = sample
                if torch.cuda.is_available():
                    data = data.cuda()

                prediction, boundary, _, aff_cup, aff_disc = model_eval(data)
                #prediction, _, feature, aff_cup, aff_disc

                bg_label = gt_cup[0].cuda(non_blocking=True)
                fg_label = gt_cup[1].cuda(non_blocking=True)
                neg_label = gt_cup[2].cuda(non_blocking=True)


                bg_count = torch.sum(bg_label) + 1e-5
                fg_count = torch.sum(fg_label) + 1e-5
                neg_count = torch.sum(neg_label) + 1e-5

                bg_loss = torch.sum(- bg_label * torch.log(aff_cup + 1e-5)) / bg_count
                fg_loss = torch.sum(- fg_label * torch.log(aff_cup + 1e-5)) / fg_count
                neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff_cup)) / neg_count

                loss_cup = bg_loss/4 + fg_loss/4 + neg_loss/2

                avg_meter_cup.add({
                    'loss': loss_cup.item(),
                    'bg_loss': bg_loss.item(), 'fg_loss': fg_loss.item(), 'neg_loss': neg_loss.item(),
                    'bg_cnt': bg_count.item(), 'fg_cnt': fg_count.item(), 'neg_cnt': neg_count.item()
                })
                #disc
                bg_label = gt_disc[0].cuda(non_blocking=True)
                fg_label = gt_disc[1].cuda(non_blocking=True)
                neg_label = gt_disc[2].cuda(non_blocking=True)

                bg_count = torch.sum(bg_label) + 1e-5
                fg_count = torch.sum(fg_label) + 1e-5
                neg_count = torch.sum(neg_label) + 1e-5

                bg_loss = torch.sum(- bg_label * torch.log(aff_disc + 1e-5)) / bg_count
                fg_loss = torch.sum(- fg_label * torch.log(aff_disc + 1e-5)) / fg_count
                neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff_disc)) / neg_count

                loss_disc = bg_loss/4 + fg_loss/4 + neg_loss/2

                avg_meter_disc.add({
                    'loss': loss_disc.item(),
                    'bg_loss': bg_loss.item(), 'fg_loss': fg_loss.item(), 'neg_loss': neg_loss.item(),
                    'bg_cnt': bg_count.item(), 'fg_cnt': fg_count.item(), 'neg_cnt': neg_count.item()
                })
        print(  '***********cup',
                'loss:%.4f %.4f %.4f %.4f' % avg_meter_cup.get('loss', 'bg_loss', 'fg_loss', 'neg_loss'),
                'cnt:%.0f %.0f %.0f' % avg_meter_cup.get('bg_cnt', 'fg_cnt', 'neg_cnt')
                )
        print(  '***********disc',
                'loss:%.4f %.4f %.4f %.4f' % avg_meter_disc.get('loss', 'bg_loss', 'fg_loss', 'neg_loss'),
                'cnt:%.0f %.0f %.0f' % avg_meter_disc.get('bg_cnt', 'fg_cnt', 'neg_cnt')
                )

        avg_meter_cup.pop()
        avg_meter_disc.pop()        
        
        model.train()

    torch.save({
                'model_state_dict': model.state_dict(),
                }, './log/sim_learn_D2.pth.tar')
