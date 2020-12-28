"""
@author Joaquin Aliaga Gonzalez
@email joaliaga.g@gmail.com
@create date 2020-12-28 10:47:18
@modify date 2020-12-28 11:00:26
@desc Check and clean dataframe for bioPDSN training, trying to read imgs and delete the ones that can't be readed
"""

import pandas as pd 
import cv2
import numpy as np
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
import os

class CheckDataset(Dataset):
    def __init__(self, dataFrame,root,input_size=[112,112]):
        self.dataFrame = dataFrame
        self.root = root
        
        self.transformations = Compose([
            ToPILImage(),
            Resize((input_size[0], input_size[1])),
            ToTensor(), # [0, 1]
        ])
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        
        row = self.dataFrame.iloc[key]
        try:
            source = cv2.imdecode(np.fromfile(self.root+row['source'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            target = cv2.imdecode(np.fromfile(self.root+row['target'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            
            source: self.transformations(source)
            target: self.transformations(target)
            clase: tensor([row['id_class']], dtype=long) # pylint: disable=not-callable
            dic = {}
        except Exception as e:
            print("Exception: ",e)
            dic = {
            'source': self.root+row['source'],
            'target': self.root+row['target'],
            #'negative': self.transformations(negative),
            'class': tensor([row['id_class']], dtype=long), # pylint: disable=not-callable
            }
        return dic
    
    def __len__(self):
        return len(self.dataFrame.index)


if __name__ == "__main__":
    


