import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.aspp import build_aspp
from networks.decoder import build_decoder
from networks.backbone import build_backbone

from tool import pyutils
import sys
import torch.sparse as sparse

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, image_res=512, radius=4):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.aff_cup = torch.nn.Conv2d(256, 256, 1, bias=False)
        self.aff_disc = torch.nn.Conv2d(256, 256, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.aff_cup.weight, gain=4)
        torch.nn.init.xavier_uniform_(self.aff_disc.weight, gain=4)
        self.bn_cup = BatchNorm(256)
        self.bn_disc = BatchNorm(256)
        self.bn_cup.weight.data.fill_(1)
        self.bn_cup.bias.data.zero_()
        self.bn_disc.weight.data.fill_(1)
        self.bn_disc.bias.data.zero_()


        self.from_scratch_layers = [self.aff_cup, self.aff_disc, self.bn_cup, self.bn_disc]
        
        self.predefined_featuresize = int(image_res//16)
        self.ind_from, self.ind_to = pyutils.get_indices_of_pairs(radius=radius, size=(self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, to_dense=False):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        feature = x #torch.Size([8, 256, 32, 32])
        x1, x2, feature_last = self.decoder(x, low_level_feat)

        x2 = F.interpolate(x2, size=input.size()[2:], mode='bilinear', align_corners=True)
        x1 = F.interpolate(x1, size=input.size()[2:], mode='bilinear', align_corners=True)
        
        f_cup = F.relu(self.bn_cup(self.aff_cup(feature)))###bn
        f_disc = F.relu(self.bn_disc(self.aff_disc(feature)))

        if f_cup.size(2) == self.predefined_featuresize and f_cup.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            print('featuresize error')
            sys.exit()

        f_cup = f_cup.view(f_cup.size(0), f_cup.size(1), -1)

        ff = torch.index_select(f_cup, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(f_cup, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff_cup = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))

        if to_dense:
            aff_cup = aff_cup.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = f_cup.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_cup = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                      torch.cat([aff_cup, torch.ones([area]), aff_cup])).to_dense().cuda()


        f_disc = f_disc.view(f_disc.size(0), f_disc.size(1), -1)

        ff = torch.index_select(f_disc, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(f_disc, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff_disc = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))

        if to_dense:
            aff_disc = aff_disc.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = f_disc.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_disc = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                      torch.cat([aff_disc, torch.ones([area]), aff_disc])).to_dense().cuda()

        return x1, x2, feature_last, aff_cup, aff_disc

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_scratch_parameters(self):
        groups = []
        
        for param in self.parameters():
            param.requires_grad = False

        for m in self.modules():
            
            if m in self.from_scratch_layers:
                
                groups.append(m.weight)
                m.weight.requires_grad = True
                if isinstance(m, SynchronizedBatchNorm2d):
                    groups.append(m.bias)
                    m.bias.requires_grad = True
                
                #groups.append(m.bias)###
                


        return groups

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


