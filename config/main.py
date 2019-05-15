import os
import sys
import argparse
import csv
import numpy as np
import time
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import settings

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)

from datasets import HAM10000_triplet_dataset, HAM10000_eval_dataset
import itertools
import solver
from models import FeatureGenerator, IdClassifier, FeatureEmbedder, Baseline
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=settings.device_id

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        ##################################### Import models ###########################
        self.feature_generator = Baseline(last_stride=1, model_path='/home/yehr/.torch/models/resnet50-19c8e357.pth')
        self.feature_embedder = FeatureEmbedder(2048)
        self.id_classifier = IdClassifier()

        if torch.cuda.is_available():
            self.feature_generator.cuda()
            self.feature_embedder.cuda()
            self.id_classifier.cuda()
        self.feature_generator = nn.DataParallel(self.feature_generator, device_ids=range(settings.num_gpu))
        self.feature_embedder = nn.DataParallel(self.feature_embedder, device_ids=range(settings.num_gpu))
        self.id_classifier = nn.DataParallel(self.id_classifier, device_ids=range(settings.num_gpu))


        ############################# Get Losses & Optimizers #########################

        self.criterion_triplet_l2 = torch.nn.TripletMarginLoss(margin = settings.triplet_margin)
        self.criterion_triplet = torch.nn.CosineEmbeddingLoss(margin = settings.triplet_margin)
        #self.criterion_identity = CrossEntropyLabelSmoothLoss(settings.num_classes, epsilon=0.1) #torch.nn.CrossEntropyLoss()
        self.criterion_identity = torch.nn.CrossEntropyLoss()

        opt_models = [self.feature_generator,
                      self.feature_embedder,
                      self.id_classifier]

        def make_optimizer(opt_models):
            train_params = []

            for opt_model in opt_models:
                for key, value in opt_model.named_parameters():
                    if not value.requires_grad:
                        continue
                    lr = settings.BASE_LR
                    weight_decay = settings.WEIGHT_DECAY
                    if "bias" in key:
                        lr = settings.BASE_LR * settings.BIAS_LR_FACTOR
                        weight_decay = settings.WEIGHT_DECAY_BIAS
                    train_params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


            optimizer = torch.optim.Adam(train_params)
            return optimizer


        self.optimizer_G = make_optimizer(opt_models)
        self.epoch_count = 0
        self.step = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.writers = {}
        self.dataloaders = {}

        self.sche_G = solver.WarmupMultiStepLR(self.optimizer_G, milestones=settings.iter_sche, gamma=0.1) # default setting 

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]


    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(name + '/' + k, v, self.step)


        out['G_lr'] = self.optimizer_G.param_groups[0]['lr']
        #out['D_lr'] = self.optimizer_D.param_groups[0]['lr']
        out['step'] = self.step
        out['eooch_count'] = self.epoch_count
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'feature_generator': self.feature_generator.state_dict(),
            'feature_embedder': self.feature_embedder.state_dict(),
            'id_classifier': self.id_classifier.state_dict(),
            'clock': self.step,
            'epoch_count': self.epoch_count,
            'opt_G': self.optimizer_G.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            print('load checkpoint: %s' %ckp_path)
        except FileNotFoundError:
            return

        self.feature_generator.load_state_dict(obj['feature_generator'])
        self.feature_embedder.load_state_dict(obj['feature_embedder'])
        self.id_classifier.load_state_dict(obj['id_classifier'])
        self.optimizer_G.load_state_dict(obj['opt_G'])
        self.step = obj['clock']
        self.epoch_count = obj['epoch_count']
        self.sche_G.last_epoch = self.step


    def inf_batch(self, batch):
        anchor_img, positive_img, negative_img, anchor_label = batch

        if torch.cuda.is_available():
            anchor_img = anchor_img.cuda()
            positive_img = positive_img.cuda()
            negative_img = negative_img.cuda()
            anchor_label = anchor_label.cuda()

        def cal_fea(x):
            feat = self.feature_generator(x)
            return self.feature_embedder(feat)

        anchor_rgb_features = cal_fea(anchor_img)
        positive_rgb_features = cal_fea(positive_img)
        negative_rgb_features = cal_fea(negative_img)

        l2triplet_loss_rgb = self.criterion_triplet_l2(anchor_rgb_features,
                                                       positive_rgb_features, negative_rgb_features)

        triplet_loss = l2triplet_loss_rgb

        predicted_id_rgb = self.id_classifier(anchor_rgb_features)

        identity_loss = self.criterion_identity(predicted_id_rgb, anchor_label) 

        return triplet_loss, identity_loss



def run_train_val(ckp_name='ckp_latest'):
    sess = Session()
    sess.load_checkpoints(ckp_name)

    sess.tensorboard('train_stats')
    sess.tensorboard('val_stats')

    ######################## Get Datasets & Dataloaders ###########################

    train_dataset = HAM10000_triplet_dataset(transforms_list=settings.transforms_list)

    def get_train_dataloader():
        return iter(DataLoader(HAM10000_triplet_dataset(transforms_list=settings.transforms_list), batch_size=settings.train_batch_size, shuffle=True,num_workers=settings.num_workers, drop_last = True))

    transform_test = settings.test_transforms_list
    def get_val_dataloader():
        return iter(DataLoader(HAM10000_triplet_dataset(transforms_list=transform_test, mode='test'), batch_size=settings.train_batch_size, shuffle=True,num_workers=settings.num_workers, drop_last = True))

    train_dataloader = get_train_dataloader()
    val_dataloader = get_val_dataloader()

    while sess.step < settings.iter_sche[-1]:
        sess.sche_G.step()
        sess.feature_generator.train()
        sess.feature_embedder.train()
        sess.id_classifier.train()

        try:
            batch_t = next(train_dataloader)
        except StopIteration:
            train_dataloader = get_train_dataloader()
            batch_t = next(train_dataloader)
            sess.epoch_count += 1

        triplet_loss, identity_loss = sess.inf_batch(batch_t)
        loss_t = identity_loss
        sess.optimizer_G.zero_grad() 
        loss_t.backward()
        sess.optimizer_G.step()

        sess.write('train_stats', {'loss_t': loss_t,
                                   'triplet_loss': triplet_loss,
                                   'identity_loss': identity_loss
        })



        if sess.step % int(settings.latest_steps) == 0:
            sess.save_checkpoints('ckp_latest')
            sess.save_checkpoints('ckp_latest_backup')

        if sess.step % settings.val_step ==0:
            sess.feature_generator.eval()
            sess.feature_embedder.eval()
            sess.id_classifier.eval()

            try:
                batch_v = next(val_dataloader)
            except StopIteration:
                val_dataloader = get_val_dataloader()
                batch_v = next(val_dataloader)

            triplet_loss_v, identity_loss_v = sess.inf_batch(batch_v)
            loss_v = triplet_loss_v + identity_loss_v

            sess.write('val_stats', {'loss_v': loss_v,
                                     'triplet_loss_v': triplet_loss_v,
                                     'identity_loss_v': identity_loss_v
            })

        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints('ckp_step_%d' % sess.step)
            logger.info('save model as ckp_step_%d' % sess.step)
        sess.step += 1


def run_test(arg):
    sess = Session()
    if arg == 'all':
        models = sorted(os.listdir('../models/'))
        csvfile = open('all_test_results.csv', 'w')
        writer = csv.writer(csvfile)

        writer.writerow(['ckp_name', 'T1', 'T5'])

        for mm in models:
            result = test_ckp(mm, sess)
            writer.writerow(result)

        csvfile.close()

    else:
        test_ckp(arg, sess)


def test_ckp(ckp_name, sess):
    sess.load_checkpoints(ckp_name)

    transform_test = settings.test_transforms_list

    test_loader = DataLoader(
        HAM10000_eval_dataset(transforms_list=transform_test, data_split='test'),
        batch_size=settings.val_batch_size, shuffle=False, num_workers=settings.val_workers,
        drop_last=False,
    )

    sess.feature_generator.eval()
    sess.feature_embedder.eval()
    sess.id_classifier.eval()
    top1_r, top5_r = cal_performance(nn.Sequential(sess.feature_generator, sess.feature_embedder, sess.id_classifier), test_loader)

    logger.info('Test for model {}, Test R1: {}, R5: {}.'.format(ckp_name, top1_r, top5_r))

    return [ckp_name, top1_r, top5_r]

def cal_performance(model, loader):
    sample_no = 0
    TOP1 = 0
    TOP5 = 0
    for batch in loader:
        bs = len(batch[0])
        inp, y = batch
        inp, y = inp.cuda(), y.cuda()
        predicted = model(inp)
        TOP1 += (y == predicted.max(dim=1).indices).sum()
        y = y[:,None]
        TOP5 += (predicted.argsort(dim=1)[:,-5:].float() == y.float()).sum()

        sample_no += bs

    TOP1_R = TOP1.float() / sample_no * 100
    TOP5_R = TOP5.float() / sample_no * 100
    print('sample_no: ')
    print(sample_no)
    print('top1, top5: ')
    print(TOP1)
    print(TOP5)
    return TOP1_R, TOP5_R

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', default='train')
    parser.add_argument('-m', '--model', default='ckp_latest')

    args = parser.parse_args(sys.argv[1:])

    if args.action == 'train':
        run_train_val(args.model)
    elif args.action == 'test':
        run_test(args.model)

