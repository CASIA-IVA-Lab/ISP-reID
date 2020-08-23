# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .dukemtmcreid import DukeMTMCreID
from .occluded_dukemtmcreid import OccludedDukeMTMCreID
from .market1501 import Market1501
from .dataset_loader import ImageDataset, ImageDataset_train
from .cuhk03_np_labeled import CUHK03_NP_labeled
from .cuhk03_np_detected import CUHK03_NP_detected

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'occluded_dukemtmc': OccludedDukeMTMCreID,
    'cuhk03_np_labeled': CUHK03_NP_labeled,
    'cuhk03_np_detected': CUHK03_NP_detected,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
