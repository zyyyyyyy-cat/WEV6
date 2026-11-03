from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'transformer'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.PCA_LOSS_WEIGHT = 0.0
_C.MODEL.GPD_MOMENTUM = 0.2
# GPD variant: 'single' (original UAD), 'subcenter' (GPD-SC), or
# 'hierarchical' (HPD: shared identity prototype + modality sub-centers).
_C.MODEL.GPD_TYPE = 'single'
# Number of sub-centers per identity for GPD-SC/HPD (3 = RGB/NIR/TIR).
_C.MODEL.GPD_SUB_CENTERS = 3
# Temperature for all GPD variants.
_C.MODEL.GPD_TAU = 0.03
# Count-aware EMA for sub-center updates.
_C.MODEL.GPD_COUNT_EMA = False
_C.MODEL.GPD_EMA_MIN = 0.1
_C.MODEL.GPD_EMA_MAX = 0.5
# Initialize sub-centers with tri-modal training-set class means.
_C.MODEL.GPD_WARMSTART = False

# HPD only: scene weight alpha in loss = (1-alpha)*CE_id + alpha*CE_scene.
# alpha=1.0 is GPD-SC; alpha=0.0 is GPD with a warm-started shared prototype.
_C.MODEL.GPD_SCENE_WEIGHT = 0.75
# HPD only: EMA new-value weight for the shared identity prototype.
_C.MODEL.GPD_SHARED_MOMENTUM = 0.2
# HPD only: every N iterations, run the gradient/assignment diagnostic.
# 0 disables the diagnostic.
_C.MODEL.GPD_DIAG_PERIOD = 0

# GAC: gallery-conditioned auxiliary classifier weight. 0 disables it and
# reproduces the V2/HPD trajectory exactly. Heads are registered but unused.
_C.MODEL.AUX_WEIGHT = 0.0
_C.MODEL.AUX_TAU = 0.03
# Linear ramp: 0.5*AUX_WEIGHT at epoch 1 -> AUX_WEIGHT at AUX_RAMP_EPOCHS.
_C.MODEL.AUX_RAMP_EPOCHS = 5

# V6: class-level cross-modal alignment loss (SupCon-style).
# All-cross-modal pair-equal (RGB-NIR / RGB-TIR / NIR-TIR) on the pre-BN CLS.
# Disabled by default so old configs reproduce the base path.
_C.MODEL.CORR_LOSS = CN()
_C.MODEL.CORR_LOSS.ENABLED = False
_C.MODEL.CORR_LOSS.WEIGHT = 0.0
_C.MODEL.CORR_LOSS.TAU = 0.1
_C.MODEL.CORR_LOSS.WARMUP_EPOCHS = 10

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# _C.MODEL.PRETRAIN_HW_RATIO = 1.0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
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
_C.DATASETS.NAMES = ('MARS')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
# Ordered modality folder names under each split directory, e.g. train/<modality>/*.jpg
_C.DATASETS.MODALITIES = ('RGB', 'IR', 'Thermal')
# Sub-directory name under ROOT_DIR, e.g. 'WHU-MARS' or 'WHU-MARS_2337'
_C.DATASETS.DATASET_DIR = 'WHU-MARS'
# Filter to Ground-Daytime subset (WHU-MARS-1000-GD evaluation protocol)
_C.DATASETS.FILTER_GD = False


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'PKM'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
# V6/2337 compatibility: skip train IDs that lack any modality.  Default False
# keeps old behavior; V6 configs set True.  For complete-modality datasets
# (1000) this is a no-op.
_C.DATALOADER.SKIP_INCOMPLETE_IDS = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# Optional full-training-state checkpoint to resume from (model+optimizer+
# scheduler+scaler+bank+RNG). Empty means train from scratch.
_C.SOLVER.RESUME = ''
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

_C.SOLVER.LOSS_TYPE = "base"

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# 0 for none top_k evaluation, otherwise will only evaluate top_k results for mAP
_C.TEST.TOP_K_EVAL= 0
# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# Run 36 directed modality-platform protocols during standalone inference.
_C.TEST.DETAILED_DOMAIN_EVAL = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
