'''
author: Joaquin Aliaga
refactor from https://github.com/JadHADDAD92/covid-mask-detector
'''

import cv2
import numpy as np
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

class MaskDataset(Dataset):
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
        source = cv2.imdecode(np.fromfile(self.root+row['source'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        target = cv2.imdecode(np.fromfile(self.root+row['target'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        negative = cv2.imdecode(np.fromfile(self.root+row['negative'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return {
            'source': self.transformations(source),
            'target': self.transformations(target),
            'negative': self.transformations(negative),
            'class': tensor([row['id_class']], dtype=long), # pylint: disable=not-callable
        }
    
    def __len__(self):
        return len(self.dataFrame.index)    