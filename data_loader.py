from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import torch
import os
import random
import pandas as pd
from math import *

import lmdb
import dill as pickle
import glob


class MVTec(data.Dataset):
    def __init__(self, data_path, image_size=(512, 512), mode="train", shot=1, augment=True, random_seed=0,subset=None, percent_defect=-1, grayscale=False):
        self.data_path = data_path
        self.classes = ['tile', 'bottle', 'leather', 'toothbrush', 'screw', 'transistor', 'capsule', 'carpet', 'grid', 'hazelnut', 'pill', 'metal_nut', 'cable', 'zipper', 'wood']
        self.mode = mode
        self.grayscale = grayscale
        self.subset=subset
        self.image_size = image_size
        self.augment = augment

        self.augment_rule = {
                    "bottle": "all",
                    "cable": "none",
                    "capsule": "none",
                    "carpet": "horiz",
                    "grid": "norotate",
                    "hazelnut": "norotate",
                    "leather": "none",
                    "metal_nut": "none",
                    "pill": "none",
                    "screw": "norotate",
                    "tile": "horiz",
                    "toothbrush": "horiz",
                    "transistor": "none",
                    "wood": "norotate",
                    "zipper": "horiz"
                    }
     
        if mode == "train":
            data_list = pd.read_csv(os.path.join(data_path,'train_all.txt'), header=None)
        elif mode == 'val':
            data_list = pd.read_csv(os.path.join(data_path,'val_train_defect.txt'), header=None)
        elif mode == 'test':
            data_list = pd.read_csv(os.path.join(data_path,'val.txt'), header=None)

        if subset != None:
            data_list = data_list[data_list[0].str.contains(subset)]

        self.data_list = list(data_list[0].values)   

        self.N = len(self.data_list)


        transform_augment = []
        transform_augment.append(T.RandomHorizontalFlip())
        transform_augment.append(T.RandomVerticalFlip())
        self.transform_augment_all = T.Compose(transform_augment)


        transform_augment = []
        transform_augment.append(T.RandomHorizontalFlip())
        self.transform_augment_horiz = T.Compose(transform_augment)



        transform_augment = []
        transform_augment.append(T.RandomHorizontalFlip())
        transform_augment.append(T.RandomVerticalFlip())
        self.transform_augment_norotate = T.Compose(transform_augment)


        transform_input = []
        transform_input.append(T.ToTensor())
        if not self.grayscale:
            transform_input.append(T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]))
        else:
            transform_input.append(T.Normalize(mean=[0.5],
                                     std=[0.5]))
        self.transform_input = T.Compose(transform_input)


    def __getitem__(self, index):
        image_id = self.data_list[index]

        image_id = image_id.replace("/Dataset/MVTec/","")

        image_path = os.path.join(self.data_path, image_id.replace("/Dataset/MVTec/",""))
        try:
            if self.grayscale:
                image = Image.open(image_path).convert("L").resize((self.image_size[1],self.image_size[0]))
            else:
                image = Image.open(image_path).convert("RGB").resize((self.image_size[1],self.image_size[0]))
        except:
            print(image_id)

        class_name = image_id.split("/")[1]
        class_id = int(self.classes.index(class_name))
        if self.subset is not None:
            class_id = 0
        if self.augment:

            rule = self.augment_rule[class_name]
            if rule == "all":
                image = self.transform_augment_all(image)
            elif rule == "horiz":
                image = self.transform_augment_horiz(image)
            elif rule == "norotate":
                image = self.transform_augment_norotate(image)
            elif rule == "none":
                image = image


        return self.transform_input(image), class_id, image_id, 0

    def __len__(self):
        return self.N  

class JetDataset(data.Dataset):
    def __init__(self, data_path, image_size=(128, 128), mode="train", shot=1, augment=True, random_seed=0,subset=None, percent_defect=-1, grayscale=False):
        self.data_path = data_path
        self.classes = ['R', 'C']
        self.mode = mode
        self.grayscale = grayscale
        self.subset=subset
        self.image_size = image_size
        self.augment = augment

        self.augment_rule = {
                    "R": "all",
                    "C": "all"
                    }
        if mode == "train":
            data_list = pd.read_csv(os.path.join(data_path,'train_all.csv'), header=None)
        elif mode == 'val':
            data_list = pd.read_csv(os.path.join(data_path,'val_defect.csv'), header=None)
        elif mode == 'val-test':
            data_list = pd.read_csv(os.path.join(data_path,'val.csv'), header=None)
        elif mode == 'test':
            data_list = pd.read_csv(os.path.join(data_path,'test.csv'), header=None)

        # if subset != None:
            # data_list = data_list[data_list[0].str.contains(subset)]
        self.data_list = list(data_list[0].values)   
        self.N = len(self.data_list)


        transform_augment = []
        transform_augment.append(T.RandomHorizontalFlip())
        transform_augment.append(T.RandomVerticalFlip())
        self.transform_augment_all = T.Compose(transform_augment)


        transform_augment = []
        transform_augment.append(T.RandomHorizontalFlip())
        self.transform_augment_horiz = T.Compose(transform_augment)



        transform_augment = []
        transform_augment.append(T.RandomHorizontalFlip())
        transform_augment.append(T.RandomVerticalFlip())
        self.transform_augment_norotate = T.Compose(transform_augment)


        transform_input = []
        transform_input.append(T.ToTensor())
        if not self.grayscale:
            transform_input.append(T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]))
        else:
            transform_input.append(T.Normalize(mean=[0.5],
                                     std=[0.5]))
        self.transform_input = T.Compose(transform_input)


    def __getitem__(self, index):
        image_id = self.data_list[index]
        image_id = image_id.replace("./dataset/Jet/","")
        image_path = os.path.join(self.data_path, image_id.replace("./dataset/Jet/",""))
        try:
            if self.grayscale:
                image = Image.open(image_path).convert("L").resize((self.image_size[1],self.image_size[0]))
            else:
                image = Image.open(image_path).convert("RGB").resize((self.image_size[1],self.image_size[0]))
        except:
            print("failed:", image_id)

        class_name = image_id.split("/")[1]
        class_id = int(self.classes.index(class_name))
        # if self.subset is not None:
            # class_id = 0
        if self.augment:

            rule = self.augment_rule[class_name]
            if rule == "all":
                image = self.transform_augment_all(image)
            elif rule == "horiz":
                image = self.transform_augment_horiz(image)
            elif rule == "norotate":
                image = self.transform_augment_norotate(image)
            elif rule == "none":
                image = image


        return self.transform_input(image), class_id, image_id, 0

    def __len__(self):
        return self.N  


def get_loader(data_path, image_size=128, batch_size=16, dataset='Jet', 
                    mode='train', shot=1, augment=True, num_workers=0, shuffle=False, subset=None, percent_defect=-1, grayscale=False):
    """Build and return a data loader."""

    if dataset == 'MVTec':
        dataset = MVTec(data_path,image_size=image_size, mode=mode, augment=augment, shot=shot,subset=subset,percent_defect=percent_defect, grayscale=grayscale)
    elif dataset == 'Jet':
        print(f"load Jet Dataset - {mode}")
        dataset = JetDataset(data_path,image_size=image_size, mode=mode, augment=augment, shot=shot,subset=subset,percent_defect=percent_defect, grayscale=grayscale)
        
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader
