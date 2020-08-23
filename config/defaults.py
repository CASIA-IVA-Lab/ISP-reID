from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'HRNet32'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
# The loss type of metric loss
# options:'triplet','cluster','triplet_cluster','center','range_center','triplet_center','triplet_range_center'
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# For example, if loss type is cross entropy loss + triplet loss + center loss
# the setting should be: _C.MODEL.METRIC_LOSS_TYPE = 'triplet_center' and _C.MODEL.IF_WITH_CENTER = 'yes'
_C.MODEL.IF_BIGG = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'


_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 128]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [128, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2

_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.EXTRA.PRETRAINED_LAYERS = ['*']
_C.MODEL.EXTRA.STEM_INPLANES = 64
_C.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
_C.MODEL.EXTRA.WITH_HEAD = True

_C.MODEL.EXTRA.STAGE2 = CN()
_C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
_C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [32, 64]
_C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE3 = CN()
_C.MODEL.EXTRA.STAGE3.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
_C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [32, 64, 128]
_C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE4 = CN()
_C.MODEL.EXTRA.STAGE4.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
_C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
_C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

_C.CLUSTERING = CN()
_C.CLUSTERING.AL = 'Kmeans'
_C.CLUSTERING.PERIOD = 2
_C.CLUSTERING.STOP = 101
_C.CLUSTERING.PART_NUM = 7
_C.CLUSTERING.ENHANCED = True


# training
_C.TRAIN = CN()
_C.TRAIN.IGNORE_LABEL = -1


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./data')
_C.DATASETS.PSEUDO_LABEL_SUBDIR = ('train_pseudo_labels')
_C.DATASETS.PREDICTED_GT_SUBDIR = ('bounding_box_train_parsing_pgt')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 2
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
_C.SOLVER.ALIGN_LOSS_WEIGHT = 0.01
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1

_C.SOLVER.PARSING_LOSS_WEIGHT = 0.1

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (30, 55)

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 64
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Whether use the aligned feature matching
_C.TEST.WITH_ARM = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
