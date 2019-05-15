import os
import logging
import transforms

BASE_LR = 3e-5
BIAS_LR_FACTOR = 2
WEIGHT_DECAY = 0.0005
WEIGHT_DECAY_BIAS = 0.
iter_sche = [10000, 20000, 30000, 60000]

train_batch_size = 12
val_batch_size = 8

log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest')
data_path = '/data/yehr/research/ham10000/data'
save_steps = 5000
latest_steps = 100
val_step = 200

num_workers = 4
val_workers = 4
num_gpu = 1
device_id = '2'
num_classes = 7
test_times = 10 # official setting

# for showing logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


############################# Hyper-parameters ################################

alpha = 1.0
beta = 1.0
triplet_margin = 0 #0.3 # 1.4

pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]
inp_size = [224, 224]

# transforms

transforms_list = transforms.Compose([transforms.RectScale(*inp_size),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(30),
                                      transforms.Pad(10),
                                      transforms.RandomCrop(inp_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=pixel_mean,
                                                           std=pixel_std),
])

test_transforms_list = transforms.Compose([
    transforms.RectScale(*inp_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=pixel_mean,
                         std=pixel_std)])


