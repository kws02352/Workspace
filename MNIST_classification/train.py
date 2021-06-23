# 라이브러리 추가하기
import os
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

# training에 필요한 hyperparameter 설정
lr = 1e-4
batch_size = 64
num_epoch = 10

ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# network 구축하기
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.