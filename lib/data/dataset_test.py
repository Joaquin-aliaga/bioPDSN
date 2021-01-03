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
        super(FaceDataset,self).__init__()
        self.dataFrame = dataFrame
        self.root = root
        self.input_size = input_size
        '''
        self.transformations = Compose([
            ToPILImage(),
            Resize((input_size[0], input_size[1])),
        ])
        '''

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        
        row = self.dataFrame.iloc[key]
        print("Source path: ",self.root+row['source'])
        print("Target path: ",self.root+row['target'])
        source = cv2.imread(self.root+row['source']).resize(self.input_size)
        target = cv2.imread(self.root+row['target']).resize(self.input_size)
        return {
            'source_path' : row['source'],
            'target_path' : row['target'],
            'source': source,
            'target': target,
            #'negative': self.transformations(negative),
            'class': tensor([row['id_class']], dtype=long) # pylint: disable=not-callable
        }
    
    def __len__(self):
        return len(self.dataFrame.index)    