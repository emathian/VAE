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

def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD
# Training hyperparameters
num_epochs = 50
batch_size =30 # medium: smaller batchsize
learning_rate = 1e-1

# Build training dataloader

train_dataset = CustomDataset()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Model
model= VAE()
model = model.cuda()
model_type = 'vae'
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)
best_loss = np.inf
model.train()
scheduler =  StepLR(optimizer, step_size=10, gamma=0.1)
qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
writer = SummaryWriter()
#scheduler =  StepLR(optimizer, step_size=5, gamma=0.1)
for epoch in range(num_epochs):
    print('epoch   ', epoch)
    tot_loss = list()
    if epoch > 3 and epoch <= 6:
        learning_rate = 1e-2
        optimizer.param_groups[0]['lr'] = learning_rate
    elif epoch > 6 and epoch <= 18:
        learning_rate = 1e-3
        optimizer.param_groups[0]['lr'] = learning_rate
    else:
        scheduler.step()
    for c, data in enumerate(train_dataloader):
        # ===================loading=====================
        if model_type in ['cnn', 'vae', 'resnet']:
            img = data.float().cuda()
        elif model_type in ['fcn', 'fcn_b']:
            img = data.float().cuda()
            img = img.view(img.shape[0], -1)

        # ===================forward=====================
        output = model(img)
        if model_type in ['vae']:
            loss = loss_vae(output[0], img, output[1], output[2], criterion)
        else:
            loss = criterion(output, img)
        # Tensorboard definitions
        writer.add_scalar('loss', loss.item(), epoch*len(train_dataloader)* batch_size + (c+1))

        tot_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    writer.add_image('Reconstructed Image',utils.make_grid(output[0]),epoch,dataformats = 'CHW')
    # ===================save_best====================
    mean_loss = np.mean(tot_loss)
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model, 'front_best_model_{}_Biggest_BN.pt'.format(model_type))
    # ===================log========================
    qqdm_train.set_infos({
      'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
      'loss': f'{mean_loss:.4f}',
    })
    # ===================save_last========================
    torch.save(model, 'front_last_model_{}_Biggest_BN.pt'.format(model_type))
