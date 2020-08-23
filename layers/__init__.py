# encoding: utf-8

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .cluster_loss import ClusterLoss
from .center_loss import CenterLoss
from .range_loss import RangeLoss
from .parsing_loss import CrossEntropy

def make_loss(cfg, num_classes):    # modified by gu
    
    feat_dim = 256
    big_dim = 1920
    
    #parsing loss
    parsing_criterion = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL)
    #triplet                                         
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    

    
    def loss_func(cls_score_part, cls_score_global, cls_score_fore, y_part, y_global, y_fore, part_pd_score, cls_target, part_target):
        
        loss = xent(cls_score_global, cls_target) + \
                xent(cls_score_fore, cls_target) + \
                xent(cls_score_part, cls_target) + \
                cfg.SOLVER.PARSING_LOSS_WEIGHT * parsing_criterion(part_pd_score, part_target) + \
                triplet(y_global, cls_target)[0] + \
                triplet(y_fore, cls_target)[0] + \
                triplet(y_part, cls_target)[0]
    
        return loss
                        
            
    return loss_func

    
def make_loss_with_center(cfg, num_classes):    # modified by gu
    
    feat_dim = 256
    big_dim = 1920
    
    #parsing loss
    parsing_criterion = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL)
    #triplet                                         
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    
    if cfg.MODEL.IF_WITH_CENTER == 'on':
        center_criterion_part = CenterLoss(num_classes=num_classes, feat_dim=feat_dim*(cfg.CLUSTERING.PART_NUM-1), use_gpu=True)
        if cfg.MODEL.IF_BIGG:
            center_criterion_global = CenterLoss(num_classes=num_classes, feat_dim=big_dim, use_gpu=True)
        else:
            center_criterion_global = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
        center_criterion_fore = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    
    def loss_func(cls_score_part, cls_score_global, cls_score_fore, y_part, y_global, y_fore, part_pd_score, cls_target, part_target):
        
        loss = xent(cls_score_global, cls_target) + \
                xent(cls_score_fore, cls_target) + \
                xent(cls_score_part, cls_target) + \
                cfg.SOLVER.PARSING_LOSS_WEIGHT * parsing_criterion(part_pd_score, part_target) + \
                triplet(y_global, cls_target)[0] + \
                triplet(y_fore, cls_target)[0] + \
                triplet(y_part, cls_target)[0] + \
                cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion_global(y_global, cls_target) + \
                cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion_fore(y_fore, cls_target) + \
                cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion_part(y_part, cls_target)
        return loss
                        
            
    return loss_func, center_criterion_part, center_criterion_global, center_criterion_fore

