import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F

class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase)  # person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + '_xy2uv')  # keypoints
        self.dir_K_flip = os.path.join(opt.dataroot, opt.phase + '_flip_xy2uv')  # keypoints

        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)

    def init_categories(self, pairLst):
        if self.opt.phase == 'train':
            pairs_file_train = pd.read_csv(pairLst)
        elif self.opt.phase == 'test':
            pairs_file_train = pd.read_csv(pairLst.replace('train', 'test'))
        else:
            raise NotImplementedError()

        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        # for i in range(self.size):
        #     if os.path.exists(os.path.join(self.dir_P, pairs_file_train.iloc[i]['from'])) and os.path.exists(os.path.join(self.dir_P, pairs_file_train.iloc[i]['to'])):
        #         pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
        #         self.pairs.append(pair)
        # self.size = len(self.pairs)
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)
        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size - 1)

        P1_name, P2_name = self.pairs[index]

        # if not os.path.exists('./fashion_data/trainB/'):
        #     P1_name, P2_name = self.pairs[index]

        P1_path = os.path.join(self.dir_P, P1_name)  # person 1
        BP1_path = os.path.join(self.dir_K, P1_name.replace('.jpg','.npy'))  # bone of person 1
        BP1_flip_path = os.path.join(self.dir_K_flip, P1_name.replace('.jpg','.npy'))
        # person 2 and its bone
        P2_path = os.path.join(self.dir_P, P2_name)  # person 2
        BP2_path = os.path.join(self.dir_K, P2_name.replace('.jpg','.npy'))  # bone of person 2
        BP2_flip_path = os.path.join(self.dir_K_flip, P2_name.replace('.jpg', '.npy'))

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path)  # h, w, c
        BP2_img = np.load(BP2_path)

        BP1 = torch.from_numpy(BP1_img).float()  # h, w, c
        BP1 = BP1.transpose(2, 0)  # c,w,h
        BP1 = BP1.transpose(2, 1)  # c,h,w

        BP2 = torch.from_numpy(BP2_img).float()
        BP2 = BP2.transpose(2, 0)  # c,w,h
        BP2 = BP2.transpose(2, 1)  # c,h,w

        BP1_flip_img = np.load(BP1_flip_path)  # h, w, c
        BP2_flip_img = np.load(BP2_flip_path)

        BP1_flip = torch.from_numpy(BP1_flip_img).float()  # h, w, c
        BP1_flip = BP1_flip.transpose(2, 0)  # c,w,h
        BP1_flip = BP1_flip.transpose(2, 1)  # c,h,w

        BP2_flip = torch.from_numpy(BP2_flip_img).float()
        BP2_flip = BP2_flip.transpose(2, 0)  # c,w,h
        BP2_flip = BP2_flip.transpose(2, 1)  # c,h,w

        P1 = self.transform(P1_img)
        P2 = self.transform(P2_img)

        return {'P1': P1, 'BP1': BP1,
                'BP1_flip': BP1_flip,
                'P2': P2, 'BP2': BP2,
                'BP2_flip': BP2_flip,
                'P1_path': P1_name, 'P2_path': P2_name}

    def __len__(self):
        if self.opt.phase == 'train':
            return self.size
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'


import glob

class KeyDataset_wopair(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, 'color')  # person images
        self.dir_K = os.path.join(opt.dataroot,  'densepose') # keypoints
        self.dir_K_flip = os.path.join(opt.dataroot, 'densepose_flip') # keypoints

        self.transform = get_transform(opt)


        self.data_P = sorted(
                glob.glob(os.path.join(f'{self.dir_P}', "*.jpg"))
            )

        self.data_K = sorted(
            glob.glob(os.path.join(f'{self.dir_K}', "*.npy"))
        )
        self.data_K_flip = sorted(
            glob.glob(os.path.join(f'{self.dir_K_flip}', "*.npy"))
        )

        self.size = len(self.data_P)


    def __getitem__(self, index):

        P1_path = self.data_P[index]
        P1_name = P1_path[-8:]
        BP1_path = self.data_K[index]
        BP1_flip_path = self.data_K_flip[index]
        # person 2 and its bone

        P1_img = Image.open(P1_path).convert('RGB')


        BP1_img = np.load(BP1_path)  # h, w, c
        BP1 = torch.from_numpy(BP1_img).float()  # h, w, c
        BP1 = BP1.transpose(2, 0)  # c,w,h
        BP1 = BP1.transpose(2, 1)  # c,h,w


        BP1_flip_img = np.load(BP1_flip_path)  # h, w, c
        BP1_flip = torch.from_numpy(BP1_flip_img).float()  # h, w, c
        BP1_flip = BP1_flip.transpose(2, 0)  # c,w,h
        BP1_flip = BP1_flip.transpose(2, 1)  # c,h,w

        if self.opt.fineSize != BP1_img.shape[0]:
            BP1 = F.interpolate(BP1[None], (self.opt.fineSize, self.opt.fineSize))[0]
            BP1_flip = F.interpolate(BP1_flip[None], (self.opt.fineSize, self.opt.fineSize))[0]

        P1 = self.transform(P1_img)


        return {'P1': P1, 'BP1': BP1,
                'BP1_flip': BP1_flip,'P1_path': P1_name}

    def __len__(self):
        return self.size

    def name(self):
        return 'KeyDataset_wopair'



class ThumanDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        #dataroot is Thuman/image_data/
        self.dir_P = os.path.join(opt.dataroot, '*','color')  # person images
        self.dir_K = os.path.join(opt.dataroot, '*', 'densepose') # keypoints
        self.dir_K_flip = os.path.join(opt.dataroot, '*','densepose_flip') # keypoints

        self.transform = get_transform(opt)

        self.samples =  sorted(
                glob.glob(os.path.join(opt.dataroot, '*'))
            )
        self.size = 400 #len(self.samples)

        # self.data_P = sorted(
        #         glob.glob(os.path.join(f'{self.dir_P}', "*.jpg"))
        #     )
        # self.data_K = sorted(
        #     glob.glob(os.path.join(f'{self.dir_K}', "*.npy"))
        # )
        # self.data_K_flip = sorted(
        #     glob.glob(os.path.join(f'{self.dir_K_flip}', "*.npy"))
        # )



    def __getitem__(self, index):
        sample_path = self.samples[index]
        #dir_path = os.path.dirname(sample_path)
        color_paths = sorted(glob.glob(os.path.join(sample_path, 'color', "*.jpg")))
        densepose_paths = sorted(glob.glob(os.path.join(sample_path, 'densepose', "*.npy")))
        densepose_flip_paths = sorted(glob.glob(os.path.join(sample_path, 'densepose_flip', "*.npy")))


        image_num_list =  list(range(360))
        random.shuffle(image_num_list)
        image_num1 = image_num_list[0]
        image_num2 = image_num_list[1]
        P1_name = os.path.basename(color_paths[image_num1])
        P2_name = os.path.basename(color_paths[image_num2])
        P1_path = color_paths[image_num1]
        BP1_path = densepose_paths[image_num1]
        BP1_flip_path = densepose_flip_paths[image_num1]
        P2_path = color_paths[image_num2]
        BP2_path = densepose_paths[image_num2]
        BP2_flip_path = densepose_flip_paths[image_num2]


        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path)  # h, w, c
        BP2_img = np.load(BP2_path)

        BP1 = torch.from_numpy(BP1_img).float()  # h, w, c
        BP1 = BP1.transpose(2, 0)  # c,w,h
        BP1 = BP1.transpose(2, 1)  # c,h,w

        BP2 = torch.from_numpy(BP2_img).float()
        BP2 = BP2.transpose(2, 0)  # c,w,h
        BP2 = BP2.transpose(2, 1)  # c,h,w

        BP1_flip_img = np.load(BP1_flip_path)  # h, w, c
        BP2_flip_img = np.load(BP2_flip_path)

        BP1_flip = torch.from_numpy(BP1_flip_img).float()  # h, w, c
        BP1_flip = BP1_flip.transpose(2, 0)  # c,w,h
        BP1_flip = BP1_flip.transpose(2, 1)  # c,h,w

        BP2_flip = torch.from_numpy(BP2_flip_img).float()
        BP2_flip = BP2_flip.transpose(2, 0)  # c,w,h
        BP2_flip = BP2_flip.transpose(2, 1)  # c,h,w

        P1 = self.transform(P1_img)
        P2 = self.transform(P2_img)

        if self.opt.fineSize != BP1_img.shape[0]:
            BP1 = F.interpolate(BP1[None], (self.opt.fineSize, self.opt.fineSize))[0]
            BP1_flip = F.interpolate(BP1_flip[None], (self.opt.fineSize, self.opt.fineSize))[0]

            BP2 = F.interpolate(BP2[None], (self.opt.fineSize, self.opt.fineSize))[0]
            BP2_flip = F.interpolate(BP2_flip[None], (self.opt.fineSize, self.opt.fineSize))[0]


        return {'P1': P1, 'BP1': BP1,
                'BP1_flip': BP1_flip,
                'P2': P2, 'BP2': BP2,
                'BP2_flip': BP2_flip,
                'P1_path': P1_name, 'P2_path': P2_name}
    def __len__(self):
        return self.size

    def name(self):
        return 'Thuman'