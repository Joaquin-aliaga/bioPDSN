"""
@author Joaquin Aliaga Gonzalez
@email joaliaga.g@gmail.com
@create date 2021-01-01 17:08:08
@modify date 2021-01-02 12:43:46
@desc [description]
"""

#from lib.models.resnet import Resnet
#from lib.models.layer import MarginCosineProduct
from lib.data.dataset import MaskDataset
from lib.Biopdsn import BioPDSN
from facenet_pytorch import MTCNN

from tqdm import tqdm

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os

class FaceVerificator(nn.Module):
    def __init__(self,args):
        super(FaceVerificator,self).__init__()
        self.device = args.device
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
            If False, the face with the highest detection probability is returned.
                (default: {True})
        keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
            select_largest parameter. If a save_path is specified, the first face is saved to that
            path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
        '''
        self.post_process = False if args.post_process == 0 else True
        self.mtcnn = MTCNN(image_size=self.imageShape[1], device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        select_largest=False, post_process=self.post_process)
        
        #model args
        self.model = BioPDSN(args)
        print("Loading model weights (trained)...")
        self.model.load_state_dict(torch.load(args.model_weights)['state_dict'], strict=False)
        print("Model weights loaded!")
        self.model = self.model.to(self.device)
        self.model.eval()

        #prepare data
        self.prepare_data()
        self.dataloader = self.test_dataloader()

    def get_face(self,img):
        #img is a torch.tensor with shape [N,C,H,W]
        #mtcnn needs [N,H,W,C]
        #Faces detection
        face_matches, probs = self.mtcnn.detect(img.permute(0,2,3,1))
        print("Face matches type: ", type(face_matches))
        print("Face matches shape: ",face_matches.shape)
        if (face_matches is not None):
            #crop face
            return self.mtcnn(img.permute(0,2,3,1))
        else:
            return face_matches
    def get_embeddings(self,source,target):
        source = self.get_face(source)
        target = self.get_face(target)

        if(source is None or target is None):
            fc = None
            fc_occ = None
        else:
            _, _, fc, fc_occ = self.model(source,target)
            
        return fc, fc_occ
        
    #return face verification confidence
    def forward(self,source,target):
        emb_source, emb_target = self.get_embeddings(source,target)

        if(emb_source is None or emb_target is None):
            return 0.0
        else:
            sim = self.cos_sim(emb_source,emb_target)
            return sim

    def prepare_data(self):
        self.testDF = pd.read_pickle(self.dfPath)
        root = os.getcwd()+'/lib/data/'
        print("Dataset shape:",self.testDF.shape)
        self.testDF = MaskDataset(self.testDF,root,input_size=[1280,960])

    def test_dataloader(self):
        return DataLoader(self.testDF, batch_size=self.batch_size, num_workers=self.num_workers,drop_last=False)

    def test(self):
        with torch.set_grad_enabled(False):
        
            for batch_idx,batch in enumerate(self.dataloader):
                print("Batch type: ",type(batch))
                #print("Batch size: ",batch.size)
                # Transfer to GPU
                #batch = batch.to(self.device)
                step_output = self.test_step(batch,batch_idx)
                self.output = pd.concat([self.output,step_output])
            
        return self.output

    def test_step(self, batch, batch_idx):
        sources, targets, labels = batch['source'], batch['target'],batch['class']
        sources_path, targets_path = batch['source_path'], batch['target_path']

        print("Sources type: ",type(sources))
        print("Sources shape: ",sources.shape)

        sims = self(sources,targets)

        step_output = pd.DataFrame()

        for (source_path,target_path,label,sim) in zip(sources_path,targets_path,labels,sims):

            step_output = step_output.append({
                'source': source_path,
                'target': target_path,
                'class': label,
                'similarity': sim

            }, ignore_index=True)
        
        return step_output
        
        



        
