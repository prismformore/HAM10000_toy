from __future__ import print_function, absolute_import
import glob
import settings
import random
import os
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave
import random
import time
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class HAM10000_triplet_dataset(Dataset):

    def __init__(self, data_path = settings.data_path, transforms_list=None, mode='train'):

        self.mode = mode
        if mode == 'train':
            self.img_file = 'generated_train_set.csv'
        elif mode == 'test':
            self.img_file = 'test_set.csv'

        self.data_path = data_path
        file_path = os.path.join(data_path, self.img_file)
        self.img_set_df = pd.read_csv(file_path)

        self.transform = transforms_list

        # self.img_set = []
        # count = 0
        # with open(file_path) as csvfile:
        #     csv_reader = csv.reader(csvfile, delimiter=',')
        #     for row in csv_reader:
        #         count += 1
        #         if count != 1:
        #             self.img_set.append(row)

        # count -= 1

    def __getitem__(self, index):

        anchor_sample = self.img_set_df.loc[index]['image_id']
        anchor_label = self.img_set_df.loc[index]['class']
        _img_set_df = self.img_set_df.drop([index])
        positive_sample =  _img_set_df[_img_set_df['class'].isin([anchor_label])].sample(n=1)['image_id'].values[0]
        negative_sample =  _img_set_df[~_img_set_df['class'].isin([anchor_label])].sample(n=1)['image_id'].values[0]
        anchor_img = Image.open(os.path.join(self.data_path, 'images', anchor_sample + '.jpg'))
        positive_img = Image.open(os.path.join(self.data_path, 'images', positive_sample + '.jpg'))
        negative_img = Image.open(os.path.join(self.data_path, 'images', negative_sample + '.jpg'))

        anchor_img = self.padding(anchor_img)
        positive_img = self.padding(positive_img)
        negative_img = self.padding(negative_img)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)


        return anchor_img, positive_img, negative_img, anchor_label

    def padding(self, img):
        w, h = img.size
        if h > w:
            pad_h = 0
            pad_w = int((h-w)/2)
        else:
            pad_h = int((w-h)/2)
            pad_w = 0

        pad = transforms.Pad((pad_w, pad_h), padding_mode='edge')
        return pad(img)

    def __len__(self):
        return len(self.img_set_df)

class HAM10000_eval_dataset(Dataset):
    def __init__(self, data_path = settings.data_path, transforms_list=None, data_split='test'):

        if data_split == 'train':
            self.img_file = 'train_set.csv'
        elif data_split == 'test':
            self.img_file = 'test_set.csv'

        self.data_path = data_path
        file_path = os.path.join(data_path, self.img_file)
        self.img_set_df = pd.read_csv(file_path)

        self.transform = transforms_list

        # self.img_set = []
        # count = 0
        # with open(file_path) as csvfile:
        #     csv_reader = csv.reader(csvfile, delimiter=',')
        #     for row in csv_reader:
        #         count += 1
        #         if count != 1:
        #             self.img_set.append(row)

        # count -= 1

    def __getitem__(self, index):

        anchor_sample = self.img_set_df.loc[index]['image_id']
        anchor_label = self.img_set_df.loc[index]['class']

        anchor_img = Image.open(os.path.join(self.data_path, 'images', anchor_sample + '.jpg'))

        anchor_img = self.padding(anchor_img)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)

        return anchor_img, anchor_label

    def __len__(self):
        return len(self.img_set_df)

    def padding(self, img):
        w, h = img.size
        if h > w:
            pad_h = 0
            pad_w = int((h-w)/2)
        else:
            pad_h = int((w-h)/2)
            pad_w = 0

        pad = transforms.Pad((pad_w, pad_h), padding_mode='edge')
        return pad(img)
    

if __name__ == "__main__":
    triplet_set = HAM10000_triplet_dataset()
    for i in range(100):
        print(triplet_set[i])

    eval_set = HAM10000_eval_dataset()
    for i in range(100):
        print(eval_set[i])


