from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
import os
from fastai.vision import *
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

        sims = self.compare_faces(sources,targets)

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
                embeddings_source = self.resnet(source)
                embeddings_target = self.resnet(target)
                
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
    
    def get_face_detections(self, image):
        face_detection = FaceDetection()
        
        face_detection.rotated = False
        width, height = image.size
        if(width > height):
            image = image.rotate(90)
            face_detection.rotated = True
        
        #Faces detection
        face_matches, probs = self.mtcnn.detect(image)

        face_detection.detections = []
        face_detection.face_count = 0
        
        if(face_matches is not None):
            face_detection.face_count = face_matches.shape[0]
            for face in face_matches:
                if(face_detection.rotated):
                    detection = {
                            'bounding_box': {
                                'height': int(face[2] - face[0]),
                                'left': int(face[0]),
                                'top': int(face[1]),
                                'width': int(face[3] - face[1])
                            },
                    }
                    face_detection.detections.append(detection)
                else:
                    detection = {
                            'bounding_box': {
                                'height': int(face[3] - face[1]),
                                'left': int(face[0]),
                                'top': int(face[1]),
                                'width': int(face[2] - face[0])
                            },
                    }
                    face_detection.detections.append(detection)

        #Face Mask Detection
        img_tensor = T.ToTensor()(image)
        pred_class, pred_idx, outputs = self.face_mask_learn.predict(Image(img_tensor))

        if(str(pred_class) == 'mask'):
            face_detection.face_mask = True
        else:
            face_detection.face_mask = False
        
        

        return face_detection, image



    
    def compare_faces(self, source_image, target_image):  
        #Faces detection
        face_matches, probs = self.mtcnn.detect(target_image)

        if(face_matches is not None):
        
            aligned_source = []
            aligned_target = []
            x_aligned, prob = self.mtcnn(source_image, return_prob=True)
            if x_aligned is not None:
                aligned_source.append(x_aligned[0])

                x_aligned, prob = self.mtcnn(target_image, return_prob=True)
                
                if x_aligned is not None:
                    for x in x_aligned:
                        aligned_target.append(x)


                aligned_source = torch.stack(aligned_source).to(self.device)
                aligned_target = torch.stack(aligned_target).to(self.device)

                #Person embeddings
                embeddings_source = self.resnet(aligned_source).detach().cpu()
                embeddings_target = self.resnet(aligned_target).detach().cpu()

                #Calculate score
                dists = [[(e1 - e2).norm().item() for e2 in embeddings_target] for e1 in embeddings_source]
                
                min_index = dists[0].index(min(dists[0]))

                score = dists[0][min_index]

                #Face Mask Detection
                img_tensor = T.ToTensor()(target_image)
                pred_class, pred_idx, outputs = self.face_mask_learn.predict(Image(img_tensor))

                if(str(pred_class) == 'mask'):
                    face_mask = True
                else:
                    face_mask = False
                
                #AntiSpoofing Detection
                pred_class, pred_idx, outputs = self.spoofing_learn.predict(Image(img_tensor))

                quality = outputs.numpy()[0]#Score entre 0 y 1

                #print(quality)

                verification_response = {
                        'bounding_box': {
                            'height': int(face_matches[min_index][3] - face_matches[min_index][1]),
                            'left': int(face_matches[min_index][0]),
                            'top': int(face_matches[min_index][1]),
                            'width': int(face_matches[min_index][2] - face_matches[min_index][0])
                        },
                        'similarity': score,
                        'quality': str(quality),
                        'face_count': face_matches.shape[0],
                        'face_mask': face_mask,
                        'status': 'Ok',
                }
            else:
                verification_response = {
                    'bounding_box': None,
                    'similarity': None,
                    'quality': None,
                    'face_count': 0,
                    'face_mask': None,
                    'status': 'Person face (source image) not found'
                }

        else:
            verification_response = {
                    'bounding_box': None,
                    'similarity': None,
                    'quality': None,
                    'face_count': 0,
                    'face_mask': None,
                    'status': 'Person face (target image) not found'
            }
        
        return verification_response
    
    def detect_faces(self, image):  
        #working with rotated selfies
        face_detection = self.get_face_detections(image)[0]
        
        detect_response = {
            'detections': face_detection.detections, 
            'face_count': face_detection.face_count,
            'face_mask': face_detection.face_mask,
        }

        
        return detect_response

