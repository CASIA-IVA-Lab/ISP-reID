# encoding: utf-8


import glob
import re

import os.path as osp
import os
from PIL import Image
from .bases import BaseImageDataset

import numpy as np


class CUHK03_NP_labeled(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'cuhk03-np/labeled'

    def __init__(self, root='/data2/kzhu/', pseudo_label_subdir='train_mask_annotations', part_num=7, verbose=True, **kwargs):
        super(CUHK03_NP_labeled, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.pseudo_label_subdir = pseudo_label_subdir
        self.pseudo_label_dir = osp.join(self.dataset_dir, self.pseudo_label_subdir)
        self.part_num = part_num
        if not osp.exists(self.pseudo_label_dir):
            os.mkdir(self.pseudo_label_dir)
        self._check_before_run()

        train = self._process_train_dir(self.train_dir, relabel=True)
        query = self._process_test_dir(self.query_dir, relabel=False)
        gallery = self._process_test_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> cuhk03_np_labeled loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_train_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            pseudo_labels_path=osp.splitext(osp.join(self.pseudo_label_dir, osp.basename(img_path)))[0]+'.png'
            dataset.append((img_path, pid, camid, pseudo_labels_path))
        return dataset
    
    def _process_test_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            #assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid, ''))

        return dataset
