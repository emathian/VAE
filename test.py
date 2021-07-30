from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import random
import torch
import torchvision.utils as utils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torchvision.transforms as transforms

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import vq, kmeans
from dataset  import CustomDataset
from vae import VAE
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
eval_batch_size = 1

# build testing dataloader
#data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomDataset(root = "/home/mathiane/VAE/LNEN_tiles.txt", test = True)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path = 'front_best_model_vae_Biggest_BN.pt'
model_type = 'vae'
model = torch.load(checkpoint_path)
model.eval()

# prediction file
out_file = 'front_best_model_vae_Biggest_BN.csv'
out_file2 = 'Vectors2.csv'
out_filemu =  'MuDist.csv'
out_file_logvar = 'LogVar.csv'
anomality = list()
filenames = []

with torch.no_grad():
  for i, (data, filename) in enumerate(test_dataloader):
        if model_type in ['cnn', 'vae', 'resnet']:
            img = data.float().cuda()
        elif model_type in ['fcn', 'fcn_b']:
            img = data.float().cuda()
            img = img.view(img.shape[0], -1)
        else:
            img = data[0].cuda()
        output = model(img)
        if model_type in ['cnn', 'resnet', 'fcn', 'fcn_b']:
            output = output
        elif model_type in ['res_vae']:
            output = output[0]
        elif model_type in ['vae']: # , 'vqvae'
            output = output[0]
            mu =   output[1]
            logvar = output[2]
        if model_type in ['fcn','fcn_b']:
            loss = eval_loss(output, img).sum(-1)
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])
        with open(out_file, 'a') as f1:
            f1.write('{}\t{}\n'.format(filename[0], loss))
        f1.close()

        flatten = output.detach().cpu().numpy().flatten()
        flatten =  flatten.reshape(1,flatten.shape[0])
        c_pd =  pd.DataFrame(data=flatten, index=[filename[0]])
        print(c_pd)
        c_pd.to_csv(out_file2, mode='a', header=False)

        mu = mu.detach().cpu().numpy().flatten()
        mu =  flatten.reshape(1,mu.shape[0])
        c_pd =  pd.DataFrame(data=mu, index=[filename[0]])
        print(c_pd)
        c_pd.to_csv(out_filemu, mode='a', header=False)


        logvar = logvar.detach().cpu().numpy().flatten()
        logvar =  logvar.reshape(1,logvar.shape[0])
        c_pd =  pd.DataFrame(data=logvar, index=[filename[0]])
        print(c_pd)
        c_pd.to_csv(out_file_logvar, mode='a', header=False)
