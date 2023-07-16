from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from glob import glob
import random

import torch
import torch.nn.functional as F
import sys


class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius-1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)

class FundusSegmentation(Dataset):

    def __init__(self,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None,
                 image_list=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        if image_list==None:
            self._base_dir = base_dir
            self.image_list = []
            self.split = split

            self.image_pool = []
            self.label_pool = []
            self.img_name_pool = []

            self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
            print(self._image_dir)
            imagelist = glob(self._image_dir + "/*.png")
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})

            self.transform = transform
            # self._read_img_into_memory()
            # Display stats
            print('Number of images in {}: {:d}'.format(split, len(self.image_list)))
        else:
            self.image_list = image_list
            self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode is 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        # _img = self.image_pool[index]
        # _target = self.label_pool[index]
        # _img_name = self.img_name_pool[index]
        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'


class FundusSegmentation_wsim(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='Domain2',
                 split='train/ROIs',
                 testid=None,
                 transform=None,
                 image_list=None,
                 pseudo='../generate_pseudo/pseudolabel_D4_new.npz',
                 radius=4
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        if image_list==None:
            self._base_dir = base_dir
            self.image_list = []
            self.split = split

            self.image_pool = []
            self.label_pool = []
            self.img_name_pool = []

            self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
            print(self._image_dir)
            imagelist = glob(self._image_dir + "/*.png")
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})

            npfilename = pseudo

            npdata = np.load(npfilename, allow_pickle=True)
            self.pseudo_label_dic = npdata['arr_0'].item()
            self.uncertain_dic = npdata['arr_1'].item()
            self.proto_pseudo_dic = npdata['arr_2'].item()

            self.transform = transform
            self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=32, radius=radius)###
            # self._read_img_into_memory()
            # Display stats
            print('Number of images in {}: {:d}'.format(split, len(self.image_list)))
        else:
            self.image_list = image_list
            self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode is 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        pseudo_label = self.pseudo_label_dic.get(_img_name)
        uncertain_map = self.uncertain_dic.get(_img_name)
        proto_pseudo = self.proto_pseudo_dic.get(_img_name)

        pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float()
        uncertain_map = torch.from_numpy(np.asarray(uncertain_map)).float()
        proto_pseudo = torch.from_numpy(np.asarray(proto_pseudo)).float()
        
        mask_0_obj = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_0_bck = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_1_obj = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_1_bck = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_0_obj[uncertain_map[0:1, ...] < 0.05] = 1.0
        mask_0_bck[uncertain_map[0:1, ...] < 0.05] = 1.0
        mask_1_obj[uncertain_map[1:, ...] < 0.05] = 1.0
        mask_1_bck[uncertain_map[1:, ...] < 0.05] = 1.0
        mask = torch.cat((mask_0_obj*pseudo_label[0:1,...] + mask_0_bck*(1.0-pseudo_label[0:1,...]), mask_1_obj*pseudo_label[1:,...] + mask_1_bck*(1.0-pseudo_label[1:,...])), dim=0)

        mask_proto = torch.zeros([2, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_proto[pseudo_label==proto_pseudo] = 1.0

        mask = mask*mask_proto
        
        pseudo_label[mask==0] = 255
        
        anco_sample = {'image': _img, 'pseudo_label': pseudo_label, 'img_name': _img_name, 'gt': _target}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)
        
        img = anco_sample['image']
        pseudo_label = anco_sample['pseudo_label']        
        img_name = anco_sample['img_name']

        gt = anco_sample['gt']

        gt_cup = self.extract_aff_lab_func(gt[0])
        gt_disc = self.extract_aff_lab_func(gt[1])
               
        label_cup = self.extract_aff_lab_func(pseudo_label[0])#torch.Size([100, 100])->torch.Size([34, 8832])
        label_disc = self.extract_aff_lab_func(pseudo_label[1])        

        #anco_sample = {'image': img, 'pseudo_label': pseudo_label, 'img_name': img_name}
            
        return img, label_cup, label_disc, img_name, gt_cup, gt_disc


    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'


class FundusSegmentation_wprob(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir,#=Path.db_root_dir('fundus')
                 dataset='Domain2',
                 split='train/ROIs',
                 testid=None,
                 transform=None,
                 image_list=None,
                 pseudo='../generate_pseudo/pseudolabel_D4_new.npz',
                
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        if image_list==None:
            self._base_dir = base_dir
            self.image_list = []
            self.split = split

            self.image_pool = []
            self.label_pool = []
            self.img_name_pool = []

            self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
            
            imagelist = glob(self._image_dir + "/*.png")
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
            
            npfilename = pseudo

            npdata = np.load(npfilename, allow_pickle=True)
            self.pseudo_label_dic = npdata['arr_0'].item()
            self.uncertain_dic = npdata['arr_1'].item()
            self.proto_pseudo_dic = npdata['arr_2'].item()
            self.prob_dic = npdata['arr_3'].item()

            self.transform = transform
            #self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=32, radius=4)###
            # self._read_img_into_memory()
            # Display stats
            print('Number of images in {}: {:d}'.format(split, len(self.image_list)))
        else:
            self.image_list = image_list
            self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode is 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]


        pseudo_label = self.pseudo_label_dic.get(_img_name)
        uncertain_map = self.uncertain_dic.get(_img_name)
        proto_pseudo = self.proto_pseudo_dic.get(_img_name)
        prob = self.prob_dic.get(_img_name)

        pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float()
        uncertain_map = torch.from_numpy(np.asarray(uncertain_map)).float()
        proto_pseudo = torch.from_numpy(np.asarray(proto_pseudo)).float()
        prob = torch.from_numpy(np.asarray(prob)).float()

        
        mask_0_obj = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_0_bck = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_1_obj = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_1_bck = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_0_obj[uncertain_map[0:1, ...] < 0.05] = 1.0
        mask_0_bck[uncertain_map[0:1, ...] < 0.05] = 1.0
        mask_1_obj[uncertain_map[1:, ...] < 0.05] = 1.0
        mask_1_bck[uncertain_map[1:, ...] < 0.05] = 1.0
        mask = torch.cat((mask_0_obj*pseudo_label[0:1,...] + mask_0_bck*(1.0-pseudo_label[0:1,...]), mask_1_obj*pseudo_label[1:,...] + mask_1_bck*(1.0-pseudo_label[1:,...])), dim=0)

        mask_proto = torch.zeros([2, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_proto[pseudo_label==proto_pseudo] = 1.0

        mask = mask*mask_proto

        pseudo_label[mask==0] = 255
        #pseudo_label = pseudo_label[0]

        anco_sample = {'image': _img, 'pseudo_label': pseudo_label, 'img_name': _img_name, 'prob':prob, 'gt': _target}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)
        
        img = anco_sample['image']
        pseudo_label = anco_sample['pseudo_label']        
        img_name = anco_sample['img_name']

        prob = anco_sample['prob']
        gt = anco_sample['gt']


        label = 1
         
        #label=1
        #anco_sample = {'image': img, 'pseudo_label': pseudo_label, 'img_name': img_name}
        
        return img, label, img_name, prob, gt

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'