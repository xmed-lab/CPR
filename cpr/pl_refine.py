import torch
import torchvision


import argparse
import importlib
import numpy as np

from torch.utils.data import DataLoader
import scipy.misc
import torch.nn.functional as F
import os.path
import imageio
import sys
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from torchvision import transforms
import matplotlib.pyplot as plt
from utils.metrics import dice_coefficient_numpy
import math
from PIL import Image

import networks.deeplabv3 as netd



os.environ['CUDA_VISIBLE_DEVICES'] = '2'
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default='./log/sim_learn_D2.pth.tar')
    parser.add_argument("--num_workers", default=0, type=int)#8
    #parser.add_argument("--alpha", default=16, type=int)
    parser.add_argument("--out_rw", type=str, default='./temp')
    parser.add_argument("--beta", default=2, type=int)#8
    parser.add_argument("--logt", default=2, type=int)#8
    parser.add_argument('--dataset', type=str, default='Domain2')
    parser.add_argument('--data-dir', default='../../../Data/Fundus/')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    parser.add_argument('--pseudo', type=str, default='../generate_pseudo/pseudolabel_D2.npz')
    parser.add_argument('--radius',type=int,default=4)

    args = parser.parse_args()
    radius = args.radius

    model = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn, image_res=800, radius=radius)
    
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, args.weights))
    checkpoint = torch.load(args.weights)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(model.predefined_featuresize)
    model.eval()

    composed_transforms_train = transforms.Compose([
        
        tr.Resize2(None,None,50,None),#512,None,32,512 #None,None,50,None      
        tr.Normalize_tf2(),
        tr.ToTensor2()
    ])
    infer_dataset = DL.FundusSegmentation_wprob(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_train, pseudo=args.pseudo)

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    dice_before_cup = 0
    dice_after_cup = 0
    dice_before_disc = 0
    dice_after_disc = 0
    pseudo_label_dic = {}
    prob_dic = {}
    for iter, (img, _, name, prob, gt) in enumerate(infer_data_loader):

        name = name[0]
        #print(name)

        orig_shape = img.shape

        prob_upsample = F.interpolate(prob, size=(img.shape[2], img.shape[3]), mode='bilinear')
        #prob_upsample = prob_upsample.squeeze(0)
        prob_upsample = (prob_upsample>0.75).float()
        
        dice_prob_cup = dice_coefficient_numpy(prob_upsample[:,0], gt[:,0])
        dice_prob_disc = dice_coefficient_numpy(prob_upsample[:,1], gt[:,1])
        
        dice_before_cup += dice_prob_cup
        dice_before_disc += dice_prob_disc
        

        dheight = int(np.ceil(img.shape[2]/16))
        dwidth = int(np.ceil(img.shape[3]/16))

        cam = prob

        with torch.no_grad():
            _, _, _, aff_cup, aff_disc = model.forward(img.cuda(), True)
            aff_mat_cup = torch.pow(aff_cup, args.beta)
            aff_mat_disc = torch.pow(aff_disc, args.beta)

            trans_mat_cup = aff_mat_cup / torch.sum(aff_mat_cup, dim=0, keepdim=True)
            trans_mat_disc = aff_mat_disc / torch.sum(aff_mat_disc, dim=0, keepdim=True)

            for _ in range(args.logt):
                trans_mat_cup = torch.matmul(trans_mat_cup, trans_mat_cup)
                trans_mat_disc = torch.matmul(trans_mat_disc, trans_mat_disc)
            

            cam_vec_cup = cam[:,0].view(1,-1)
            cam_vec_disc = cam[:,1].view(1,-1)

            cam_rw_cup = torch.matmul(cam_vec_cup.cuda(), trans_mat_cup)
            cam_rw_disc = torch.matmul(cam_vec_disc.cuda(), trans_mat_disc)

            cam_rw_cup = cam_rw_cup.view(1, 1, dheight, dwidth)
            cam_rw_disc = cam_rw_disc.view(1, 1, dheight, dwidth)

            cam_rw_save_cup = torch.nn.Upsample((512, 512), mode='bilinear')(cam_rw_cup)
            cam_rw_save_disc = torch.nn.Upsample((512, 512), mode='bilinear')(cam_rw_disc)
            cam_rw = torch.stack((cam_rw_save_cup[0,0]/torch.max(cam_rw_save_cup[0,0]), cam_rw_save_disc[0,0]/torch.max(cam_rw_save_disc[0,0])))#/torch.max(cam_rw_save_cup[0,0])#/torch.max(cam_rw_save_disc[0,0])

            prob_dic[name] = cam_rw.detach().cpu().numpy()
            pseudo_label_dic[name] = (cam_rw>0.75).long().detach().cpu().numpy()
            

            cam_rw_save_cup = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw_cup)
            cam_rw_save_disc = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw_disc)
            cam_rw = torch.stack((cam_rw_save_cup[0,0]/torch.max(cam_rw_save_cup[0,0]), cam_rw_save_disc[0,0]/torch.max(cam_rw_save_disc[0,0])))#/torch.max(cam_rw_save_cup[0,0])#/torch.max(cam_rw_save_disc[0,0])
            

            pseudo_label_rw = (cam_rw>0.75).long().detach().cpu().numpy()###(0.75*torch.max(cam_rw_save))

            plt.subplot(1, 4, 4,title='after')
            plt.imshow(pseudo_label_rw[1])


            dice_cam_rw_cup = dice_coefficient_numpy(np.expand_dims(pseudo_label_rw[0],0), gt[:,0])
            dice_cam_rw_disc = dice_coefficient_numpy(np.expand_dims(pseudo_label_rw[1],0), gt[:,1])
            dice_after_cup += dice_cam_rw_cup
            dice_after_disc += dice_cam_rw_disc
            
    dice_before_cup /= len(infer_data_loader)
    dice_before_disc /= len(infer_data_loader)
    dice_after_cup /= len(infer_data_loader)    
    dice_after_disc /= len(infer_data_loader)

    print('%.4f,%.4f'%(dice_before_cup,dice_after_cup))
    print('%.4f,%.4f'%(dice_before_disc,dice_after_disc))

    if args.dataset=="Domain2":
        np.savez('./log/pseudolabel_D2_new', pseudo_label_dic, prob_dic)

    if args.dataset=="Domain4":
        np.savez('./log/pseudolabel_D4_new', pseudo_label_dic, prob_dic)

    if args.dataset=="Domain1":
        np.savez('./log/pseudolabel_D1_new', pseudo_label_dic, prob_dic)        