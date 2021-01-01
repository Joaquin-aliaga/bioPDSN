"""
@author Joaquin Aliaga Gonzalez
@email joaliaga.g@gmail.com
@create date 2021-01-01 17:08:08
@modify date 2021-01-01 20:04:41
@desc [description]
"""

from lib.models.resnet import Resnet
from lib.models.layer import MarginCosineProduct
from lib.data.dataset import MaskDataset
from lib.Biopdsn import BioPDSN
from facenet_pytorch import MTCNN

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

import torch
import torch.nn as nn

import os

class FaceVerificator(pl.LightningModule):
    def __init__(self,args):
        super(FaceVerificator,self).__init__()
        self.output = pd.DataFrame()
        
        #data args
        self.dfPath = args.dfPath
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        #CosineSimilarity function
        self.cos_sim = nn.CosineSimilarity()
    
        #mtcnn args
        self.imageShape = [int(x) for x in args.input_size.split(',')]
        '''
        select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
 |          If False, the face with the highest detection probability is returned.
 |          (default: {True})
 |      keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
 |          select_largest parameter. If a save_path is specified, the first face is saved to that
 |          path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
        '''
        self.post_process = False if args.post_process == 0 else True
        self.mtcnn = MTCNN(image_size=self.imageShape[1], device = self.device, 
        select_largest=False, keep_all=False, post_process=self.post_process)
        
        #model args
        self.model = BioPDSN(args)
        print("Loading model weights (trained)...")
        self.model.load_state_dict(torch.load(args.model_weights)['state_dict'], strict=False)
        print("Model weights loaded!")
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_faces(self,batch):
        if (type(batch) == list):
            batch = [img.resize(self.imageShape[1]) for img in batch]
        return self.mtcnn(batch)

    def get_embeddings(self,source,target):
        source = self.mtcnn(source)
        target = self.mtcnn(target)

        _, _, fc, fc_occ = self.model(source,target)
        
        return fc, fc_occ
        
    #return face verification confidence
    def forward(self,source,target):
        emb_source, emb_target = self.get_embeddings(source,target)
        sim = self.cos_sim(emb_source,emb_target)

        return sim

    def prepare_data(self):
        self.testDF = pd.read_pickle(self.dfPath)
        root = os.getcwd()+'/lib/data/'
        print("Dataset shape:",self.testDF.shape)
        self.testDF = MaskDataset(self.testDF,root,self.imageShape[-2:])

    def test_dataloader(self):
        return DataLoader(self.testDF, batch_size=self.batch_size, num_workers=self.num_workers,drop_last=False)
 
    def test_step(self, batch, batch_idx):
        sources, targets, labels = batch['source'], batch['target'],batch['class']
        sources_path, targets_path = batch['source_path'], batch['target_path']

        sims = self(sources,targets)

        for (source_path,target_path,label,sim) in zip(sources_path,targets_path,labels,sims):

            self.output = self.output.append({
                'source': source_path,
                'target': target_path,
                'class': label,
                'similarity': sim

            }, ignore_index=True)
        
        



        
