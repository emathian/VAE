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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            #Batch Normalisartion
            nn.BatchNorm2d(12, momentum=0.01),

            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            # Batch Normalisartion
            #nn.BatchNorm2d(24, momentum=0.01),

            ## bigger
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            # Batch Normalisartion
            nn.BatchNorm2d(48, momentum=0.01),


            ## Biggest (Bigger + one layer)
            nn.Conv2d(48, 96, 4, stride=2, padding=1),
               nn.ReLU(),
            # Batch Normalisartion
            nn.BatchNorm2d(96, momentum=0.01)
        )
        self.enc_out_1 = nn.Sequential(
            #nn.Conv2d(24, 48, 4, stride=2, padding=1),
            #nn.ReLU(),
            ## Batch Normalisartion
            #nn.BatchNorm2d(48, momentum=0.01),

            # bigger
            #nn.Conv2d(48, 96, 4, stride=2, padding=1),
            #nn.ReLU(),
            ## Batch Normalisartion
            #nn.BatchNorm2d(96, momentum=0.01),

            # Biggest
            nn.Conv2d(96, 192, 4, stride=2, padding=1),
            nn.ReLU(),
            # Batch Normalisartion
            nn.BatchNorm2d(192, momentum=0.01)
        )
        self.enc_out_2 = nn.Sequential(
            #nn.Conv2d(24, 48, 4, stride=2, padding=1),
            #nn.ReLU(),
            ## Batch Normalisartion
            #nn.BatchNorm2d(48, momentum=0.01),
            # bigger
            #nn.Conv2d(48, 96, 4, stride=2, padding=1),
            #nn.ReLU(),
            # Batch Normalisartion
            #nn.BatchNorm2d(96, momentum=0.01),


            # Biggest
            nn.Conv2d(96, 192, 4, stride=2, padding=1),
            nn.ReLU(),
            # Batch Normalisartion
            nn.BatchNorm2d(192, momentum=0.01)

        )
        self.decoder = nn.Sequential(
            ## Biggest
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),
            nn.ReLU(),
            # Batch Normalisartion
            nn.BatchNorm2d(96, momentum=0.01),

            ## Bigger
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            # Batch Normalisartion
            nn.BatchNorm2d(48, momentum=0.01),

            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            ## Batch Normalisartion
            nn.BatchNorm2d(24, momentum=0.01),

            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            # Batch Normalisartion
            nn.BatchNorm2d(12, momentum=0.01),

            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


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
