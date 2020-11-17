import argparse
import os
import time
import gc

import torch
import torchvision
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
cudnn.benchmark = True

import _init_paths
import models.net_sia as net_sia
import datasets.dataset_pair as dset
import layer
import utils

plt.switch_backend('agg')
configurations = {
    1: dict(
        lr = 1.0e-1,
        step_size = [7500, 15000, 22500],
        epochs = 50,
    ),
    2: dict(
        lr=1.0e-1,
        step_size=[10000, 17500, 22500],
        epochs = 50,
    )
}

# Training settings
parser = argparse.ArgumentParser(description="Pytorch CosFace")

# DATA
parser.add_argument('--root_path', type=str, default='/home/joaquin/Documentos/GeoVictoria/bioPDSN/data',
                    help='path to root path of images')
parser.add_argument('--train_list', type=str, default=None, help='path to training pair list')
parser.add_argument('--valid_list', type=str, default=None, help='path to validating pair list')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 512')
parser.add_argument('--is_gray', type=bool, default=False,
                    help='Transform input image to gray or not (default: False')

#Network
parser.add_argument('--weight_model', type=str, default='checkpoint/Mar02-00-34-21/CosFace_15_checkpoint.pth')
parser.add_argument('--weight_fc', type=str, default='checkpoint/Mar02-00-34-21/CosFace_15_checkpoint_classifier.pth')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--resume_fc', type=str, default='')
parser.add_argument('--s_weight', type=float, default=10.0)

#Classifier
parser.add_argument('--num_class', type=int, default=2622, help='number of people(class)')
parser.add_argument('--classifier_type', type=str, default='MCP',
                    help='Which classifier for train. (MCP, AL, L')

# LR policy
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default:30')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (defauñt: 0.1')
parser.add_argument('--step_size', type=list, default=None, help='lr decay step')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default:0.9)')
parser.add_argument('--weight_decay',type=float, default=5e-4, metavar='W'
                    help='weight decay (default:0.0005)')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                    help='the number of settings and hyperparameters used in training')

# Common settings
parser.add_argument('--log_interval', type=int, default=100, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='', help='path to save checkpoint')
parser.add_argument('--no_cuda', type=bool, default=False, help='disables CUDA training')
parser.add_argument('--workers', type=int, default=8, help='how many workers to load data')
parser.add_argument('--gpus', type=str , default='0')
parser.add_argument('--ngpus',type=int, default=1),
parser.add_argument('--d_name',type=str, default='')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

def main():
    # model #
    model = net_sia.LResNet50E_IR_Sia(is_gray=args.is_gray)
    model_eval = net_sia.LResNet50E_IR_Sia(is_gray=args.is_gray)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # 512 is dimension of feature
    classifier = {
        'MCP': layer.MarginCosineProduct(512, args.num_class),
        'AL' : layer.AngleLinear(512, args.num_class),
        'L' : torch.nn.Linear(512, args.num_class, bias=False)
    }[args.classifier_type]

    classifier.load_state_dict(torch.load(args.weight_fc))

    print(os.environ['CUDA_VISIBLE_DEVICES'], args.cuda)
