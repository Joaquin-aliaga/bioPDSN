'''
author: Joaquin Aliaga
refactor from https://github.com/JadHADDAD92/covid-mask-detector
'''

import cv2
import numpy as np
from PIL import Image
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
import os

class FaceDataset(Dataset):
    def __init__(self, dataFrame,root,input_size):
        self.dataFrame = dataFrame
        self.root = root
        
    def __getitem__(self, key,resize=(960,1280)):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        
        row = self.dataFrame.iloc[key]
        source = Image.open(self.root+row['source']).resize(resize)
        target = Image.open(self.root+row['target']).resize(resize)
        return {
            'source_path' : row['source'],
            'target_path' : row['target'],
            'source': source,
            'target': target,
            #'negative': self.transformations(negative),
            'class': tensor([row['id_class']], dtype=long), # pylint: disable=not-callable
        }
    
    def __len__(self):
        return len(self.dataFrame.index)    