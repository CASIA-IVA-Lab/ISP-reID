# encoding: utf-8


import torch
from torch import nn
import torch.nn.functional as F

from .backbones.cls_hrnet import HighResolutionNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, cfg, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        
        if model_name == 'HRNet32':
            self.in_planes = 256
            self.big_planes = 1920
            self.base = HighResolutionNet(cfg)
        

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.part_num = cfg.CLUSTERING.PART_NUM
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.bigG = cfg.MODEL.IF_BIGG
        self.arm = cfg.TEST.WITH_ARM

        #'bnneck':
        if self.neck == 'bnneck':
            #part
            self.bottleneck_part = nn.BatchNorm1d(self.in_planes*(self.part_num-1))
            self.bottleneck_part.bias.requires_grad_(False)  # no shift
            self.classifier_part = nn.Linear(self.in_planes*(self.part_num-1), self.num_classes, bias=False)
    
            self.bottleneck_part.apply(weights_init_kaiming)
            self.classifier_part.apply(weights_init_classifier)
                
            #global
            if self.bigG:
                self.bottleneck_global = nn.BatchNorm1d(self.big_planes)
                self.bottleneck_global.bias.requires_grad_(False)  # no shift
                self.classifier_global = nn.Linear(self.big_planes, self.num_classes, bias=False)
            else:
                print('no big global')
                self.bottleneck_global = nn.BatchNorm1d(self.in_planes)
                self.bottleneck_global.bias.requires_grad_(False)  # no shift
                self.classifier_global = nn.Linear(self.in_planes, self.num_classes, bias=False)
    
            self.bottleneck_global.apply(weights_init_kaiming)
            self.classifier_global.apply(weights_init_classifier)
            
            #fore
            self.bottleneck_fore = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_fore.bias.requires_grad_(False)  # no shift
            self.classifier_fore = nn.Linear(self.in_planes, self.num_classes, bias=False)
    
            self.bottleneck_fore.apply(weights_init_kaiming)
            self.classifier_fore.apply(weights_init_classifier)
            
            

    def forward(self, x):
        
        y_part, y_global, y_fore, clustering_feat_map, part_pd_score = self.base(x)  # (b, 2048, 1, 1)
        y_part = y_part.view(y_part.shape[0], -1)  # flatten to (bs, 2048)
        y_global = y_global.view(y_global.shape[0], -1)
        y_fore = y_fore.view(y_fore.shape[0], -1)

        
        if self.neck == 'bnneck':
            feat_part = self.bottleneck_part(y_part)
            feat_global = self.bottleneck_global(y_global)
            feat_fore = self.bottleneck_fore(y_fore)

        if self.training:
            cls_score_part = self.classifier_part(feat_part)
            cls_score_global = self.classifier_global(feat_global)
            cls_score_fore = self.classifier_fore(feat_fore)
            return cls_score_part, cls_score_global, cls_score_fore, y_part, y_global, y_fore, part_pd_score  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                if self.arm:
                    part_visible = torch.argmax(F.softmax(part_pd_score, dim=1), dim=1)
                    visible_part=[]
                    for pic in range(part_visible.shape[0]):
                        visible_part.append([int(part in part_visible[pic]) for part in range(1, self.part_num)])
                    return torch.cat((feat_global, feat_fore), 1), feat_part.view(feat_part.shape[0], self.part_num-1, self.in_planes), visible_part, clustering_feat_map
                else:
                    return torch.cat((feat_part, feat_global, feat_fore), 1), clustering_feat_map
        
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
