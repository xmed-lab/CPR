import argparse
import os
import os.path as osp
import torch.nn.functional as F

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
# from scipy.misc import imsave
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *

get_hd = True

def inference(model,test_loader):
    
    model.eval()
    model_eval = model

    val_cup_dice = 0.0;val_disc_dice = 0.0;datanum_cnt = 0.0
    cup_hd = 0.0; disc_hd = 0.0;datanum_cnt_cup = 0.0;datanum_cnt_disc = 0.0
    with torch.no_grad():
        for batch_idx, (sample) in enumerate(test_loader):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            prediction, boundary, _ = model_eval(data)
            prediction = torch.sigmoid(prediction)

            target_numpy = target.data.cpu()
            prediction = prediction.data.cpu()
            prediction[prediction>0.75] = 1;prediction[prediction <= 0.75] = 0


            cup_dice = dice_coefficient_numpy(prediction[:,0, ...], target_numpy[:, 0, ...])
            disc_dice = dice_coefficient_numpy(prediction[:,1, ...], target_numpy[:, 1, ...])

            for i in range(prediction.shape[0]):
                hd_tmp = hd_numpy(prediction[i, 0, ...], target_numpy[i, 0, ...], get_hd)
                if np.isnan(hd_tmp):
                    datanum_cnt_cup -= 1.0
                else:
                    cup_hd += hd_tmp

                hd_tmp = hd_numpy(prediction[i, 1, ...], target_numpy[i, 1, ...], get_hd)
                if np.isnan(hd_tmp):
                    datanum_cnt_disc -= 1.0
                else:
                    disc_hd += hd_tmp

            val_cup_dice += np.sum(cup_dice)
            val_disc_dice += np.sum(disc_dice)

            datanum_cnt += float(prediction.shape[0])
            datanum_cnt_cup += float(prediction.shape[0])
            datanum_cnt_disc += float(prediction.shape[0])

    val_cup_dice /= datanum_cnt
    val_disc_dice /= datanum_cnt
    cup_hd /= datanum_cnt_cup
    disc_hd /= datanum_cnt_disc
    
    print("cup: %.4f disc: %.4f avg: %.4f cup: %.4f disc: %.4f avg: %.4f" %
            (val_cup_dice, val_disc_dice, (val_cup_dice+val_disc_dice)/2.0, cup_hd, disc_hd, (cup_hd+disc_hd)/2.0))
    