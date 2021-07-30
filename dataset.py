import os
import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torchvision.transforms as transforms

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from torch.optim import Adam, AdamW

from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import vq, kmeans

from qqdm import qqdm, format_str
import pandas as pd
import pdb  # use pdb.set_trace() to set breakpoints for debugging
from sklearn import metrics
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import seaborn as sns




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root="/home/mathiane/VAE/Normal_tiles.txt", test = False):
        self.test =test
        self.root = root
        torch.manual_seed(123)
        with open(root, 'r') as f:
            content =  f.readlines()
        self.files_list = []
        for x in content:
            x =  x.strip()
            if x.find('reject') == -1:
                self.files_list.append(x)

        ## Image Transformation ##
        # High color augmntation
        # Random orientation
        self.transform = transforms.Compose([
            transforms.Resize((550,550)),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.ToTensor(),
#            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def __getitem__(self,index):
        img =  Image.open(self.files_list[index])
        if self.transform is not None:
            img_c = self.transform(img)
        if self.test == False:
            return img_c
        else:
            return img_c ,    self.files_list[index]

    def __len__(self):
        return len(self.files_list)
