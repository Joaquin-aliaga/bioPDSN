from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
import os
#from fastai.vision import *
import torchvision.transforms as T
from lib.data.dataset_test import FaceDataset
from tqdm import tqdm

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

import os



class FaceDetection():
    

    def __init__(self):
        self.rotated = None
        self.face_count = None
        self.detections = None
        self.face_mask = None



class FaceBio(nn.Module):
    def __init__(self,args):
        super(FaceBio,self).__init__()
        #self.workers = 0 if os.name == 'nt' else 4
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
        
        self.post_process = False if args.post_process == 0 else True

        self.mtcnn = MTCNN(
            image_size=self.imageShape[1], margin=0, min_face_size=80,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=self.post_process,
            device=self.device, keep_all=False,select_largest=False
        )

        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        #self.face_mask_learn = load_learner('./model', 'export.pkl')
        #self.spoofing_learn = load_learner('./model', 'spoofing_model.pkl')

        #prepare data
        self.prepare_data()
        self.dataloader = self.test_dataloader()


    def prepare_data(self):
        self.testDF = pd.read_pickle(self.dfPath)
        root = os.getcwd()+'/lib/data/'
        print("Dataset shape:",self.testDF.shape)
        self.testDF = FaceDataset(self.testDF,root,input_size=(960,1280))

    def test_dataloader(self):
        return DataLoader(self.testDF, batch_size=self.batch_size, num_workers=self.num_workers,drop_last=False)

    def test(self):

        with torch.set_grad_enabled(False):
        
            for batch_idx,batch in enumerate(tqdm(self.dataloader,desc="Running test")):
                step_output = self.test_step(batch,batch_idx)
                self.output = pd.concat([self.output,step_output])
            
        return self.output

    def test_step(self, batch, batch_idx):
        sources, targets, labels = batch['source'], batch['target'],batch['class']
        sources_path, targets_path = batch['source_path'], batch['target_path']

        sims = self(sources,targets)

        step_output = pd.DataFrame()

        for (source_path,target_path,label,sim) in zip(sources_path,targets_path,labels,sims):
            step_output = step_output.append({
                'source': source_path,
                'target': target_path,
                'class': label.item(),
                'similarity': sim

            }, ignore_index=True)
        
        return step_output
    
    def get_faces(self,img):
        bbx,prob = self.mtcnn.detect(img)
        output = []
        for i in range(bbx.shape[0]):
            if bbx[i] is not None:
                output.append(self.mtcnn.extract(img[i],bbx[i],None))
            else:
                output.append(None)
        return output

    def get_embeddings(self,source,target):
        sources = self.get_faces(source)
        targets = self.get_faces(target)
        #sources and targets are list with len = batch_size

        source_output = []
        target_output = []
        for source,target in zip(sources,targets):
            if source is not None and target is not None:
                #unsqueeze: add one dimention
                embeddings_source = self.resnet(source.unsqueeze_(0))
                embeddings_target = self.resnet(target.unsqueeze_(0))
                
                source_output.append(embeddings_source)
                target_output.append(embeddings_target)
            else:
                source_output.append(None)
                target_output.append(None)
        
        return source_output,target_output

    def forward(self,source,target):
        #source and target embeddings are list of len = batch
        embeddings_source, embeddings_target = self.get_embeddings(source,target)

        sims = []
        for embedding_s,embedding_t in zip(embeddings_source,embeddings_target):
            if embedding_s is not None and embedding_t is not None:
                #get the restulting similarity (float)
                sim = self.cos_sim(embedding_s,embedding_t).item()
            else:
                sim = 0.0
            sims.append(sim)
        return sims