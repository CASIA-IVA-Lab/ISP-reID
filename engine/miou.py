import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
#from utils.transforms import transform_parsing

LABELS = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg',
          'Right-leg', 'Left-shoe', 'Right-shoe']



def get_confusion_matrix(gt_label, pred_label, num_classes):
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix




def compute_IoU_one_cls(preds_dir, gt_dir, cls_name, part_num):
    image_list=os.listdir(gt_dir)

    confusion_matrix = np.zeros((2, 2))

    for i, im_name in enumerate(image_list):
        gt_path = os.path.join(gt_dir, im_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        pred_path = os.path.join(preds_dir, im_name)
        pred = np.asarray(PILImage.open(pred_path))

        gt = np.asarray(gt, dtype=np.int32)
        if cls_name == 'foreground':
            gt = np.int64(gt>0)
        elif cls_name == 'head':
            gt = np.int64(gt==1 )+ np.int64(gt==2) + np.int64(gt==4) + np.int64(gt==13)
        elif cls_name == 'legs':
            gt = np.int64(gt==8)+ np.int64(gt==9) + np.int64(gt==16) + np.int64(gt==17)
        elif cls_name == 'shoes':
            gt = np.int64(gt==18)+ np.int64(gt==19)
        gt = cv2.resize(gt, (32, 64), interpolation=cv2.INTER_NEAREST)
        pred = np.asarray(pred, dtype=np.int32)
        if cls_name=='foreground':
            pred = np.int64(pred>0)
        elif cls_name == 'head':
            pred = np.int64(pred==1)
        elif cls_name == 'legs' and part_num == 6:
            pred = np.int64(pred==4) #+ np.int64(pred==5)
        elif cls_name == 'legs' and part_num == 7:
            pred = np.int64(pred==4) + np.int64(pred==5)
        elif cls_name == 'shoes' and part_num == 6:
            pred = np.int64(pred==5)
        elif cls_name == 'shoes' and part_num == 7:
            pred = np.int64(pred==6)
        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, 2)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum())*100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean())*100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array*100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('IoU: %f \n' % mean_IoU)

def compute_IoU(preds_dir, gt_dir, part_num):
    
    #foreground
    print('Class \'Foreground\':')
    compute_IoU_one_cls(preds_dir, gt_dir, 'foreground', part_num)
    if part_num != 6 and part_num != 7:
        print('We only evaluate detailed human semantic parsing results for K=6 or K=7.')
    else:
        #head
        print('Class \'Head\':')
        compute_IoU_one_cls(preds_dir, gt_dir, 'head', part_num)
        #legs
        print('Class \'Legs\':')
        compute_IoU_one_cls(preds_dir, gt_dir, 'legs', part_num)
        #shoes
        print('Class \'Shoes\':')
        compute_IoU_one_cls(preds_dir, gt_dir, 'shoes', part_num)
