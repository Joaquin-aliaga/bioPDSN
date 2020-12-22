from lib.models.resnet import Resnet
from lib.models.layer import MarginCosineProduct
from facenet_pytorch import MTCNN
from lib.data.dataset_triplet import MaskDataset

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import os

class BioPDSN(pl.LightningModule):
    def __init__(self,args):
        super(BioPDSN,self).__init__()
        #data args
        self.dfPath = args.dfPath
        self.df = None
        self.trainDF = None
        self.validateDF = None
        #train args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        #self.momentum = args.momentum
        #self.weight_decay = args.weight_decay
        self.num_class = args.num_class
        #loss criterion
        self.loss_cls = nn.CrossEntropyLoss().to(self.device)
        self.L2loss = nn.MSELoss().to(self.device) 
        self.margin = args.margin
        
        #model args 
        self.imageShape = [int(x) for x in args.input_size.split(',')]
        self.features_shape = args.embedding_size
        
        #nets
        self.classifier = MarginCosineProduct(self.features_shape, self.num_class)
        '''
        if(args.use_mtcnn == "False"):
            self.use_mtcnn = False
        else:
            self.use_mtcnn = True
        if(self.use_mtcnn):
            self.mtcnn = MTCNN(image_size=self.imageShape[1], min_face_size=80, 
                            device = self.device, post_process=args.mtcnn_norm,
                            keep_all=args.keep_all)
        else:
            self.mtcnn = None
        '''
        self.resnet = Resnet(args)
        
        # Mask Generator
        self.sia = nn.Sequential(
            #nn.BatchNorm2d(filter_list[4]),
            nn.Conv2d(self.features_shape, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(self.features_shape),
            nn.BatchNorm2d(self.features_shape),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.features_shape * 7 * 7),
            #nn.Dropout(p=0),
            nn.Linear(self.features_shape * 7 * 7, self.features_shape),
            nn.BatchNorm1d(self.features_shape),
        )
        # Weight initialization
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        
        #self.freeze_layers('mtcnn')
        
    def get_parameters(self,filter=None):
        for name,param in self.named_parameters():
            if(filter is not None and not filter in name):
                print("Parametro: ",name)

    def freeze_layers(self,filter):
        for name, param in self.named_parameters():
            if filter in name:
                param.requires_grad = False
    
    def prepare_data(self):
        self.df = pd.read_pickle(self.dfPath)
        root = os.getcwd()+'/lib/data/'
        train, validate = train_test_split(self.df, test_size=0.2, random_state=42,stratify=self.df.id_class)
        self.trainDF = MaskDataset(train,root,self.imageShape[-2:])
        self.validateDF = MaskDataset(validate,root,self.imageShape[-2:])
        
    def get_faces(self,batch):
        if (type(batch) == list):
            batch = [img.resize(self.imageShape[1]) for img in batch]
        return self.mtcnn(batch)

    
    def get_features(self,batch):
        features = self.resnet.get_features(batch) #type(features) = numpy ndarray

        return features

    def forward(self,source,target,negative):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        f_clean = self.get_features(source.cpu())
        f_occ = self.get_features(target.cpu())
        f_neg = self.get_features(negative.cpu())
        
        f_clean = torch.from_numpy(f_clean).to(self.device)
        f_occ = torch.from_numpy(f_occ).to(self.device)
        f_neg = torch.from_numpy(f_neg).to(self.device)
        
        # Falta terminar esto!!

        # Begin Siamese branch
        f_diff = torch.add(f_clean,f_occ,alpha=-1.0)
        f_diff = torch.abs(f_diff)
        mask = self.sia(f_diff)

        f_diff_neg = torch.add(f_clean,f_neg,alpha=-1.0)
        f_diff_neg = torch.abs(f_diff_neg)
        mask_neg = self.sia(f_diff)

        # End Siamese branch

        f_clean_masked = f_clean * mask
        f_occ_masked = f_occ * mask
        f_neg_masked = f_neg * mask_neg
        
        fc = f_clean_masked.view(f_clean_masked.size(0), -1) #256*(512*7*6)
        fc_occ = f_occ_masked.view(f_occ_masked.size(0), -1)
        fc = self.fc(fc)
        fc_occ = self.fc(fc_occ)

        return f_clean_masked, f_occ_masked, f_neg_masked, fc, fc_occ

    def train_dataloader(self):
        return DataLoader(self.trainDF, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.validateDF, batch_size=self.batch_size, num_workers=self.num_workers,drop_last=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.lr)
    
        return optimizer
    
    def training_step(self, batch, batch_idx):
        sources, targets, negatives, labels = batch['source'], batch['target'], batch['negative'],batch['class']
        labels = labels.flatten()
        f_clean_masked, f_occ_masked, f_neg_masked,fc, fc_occ = self(sources,targets,negatives)
        
        triplet_loss = torch.max([self.L2loss(f_clean_masked,f_occ_masked) - self.L2loss(f_clean_masked,f_neg_masked) + self.margin,0])
                
        score_clean = self.classifier(fc, labels)
        loss_clean = self.loss_cls(score_clean, labels)
        
        score_occ = self.classifier(fc_occ, labels)
        loss_occ = self.loss_cls(score_occ, labels)
        
        lamb = 10
        loss = 0.5 * loss_clean + 0.5 * loss_occ + lamb * triplet_loss
        
        tensorboardLogs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboardLogs}

    def validation_step(self, batch, batch_idx):
        sources, targets, negatives, labels = batch['source'], batch['target'], batch['negative'],batch['class']
        labels = labels.flatten()
        f_clean_masked, f_occ_masked, f_neg_masked, fc, fc_occ = self(sources,targets,negatives)
        
        triplet_loss = torch.max([self.L2loss(f_clean_masked,f_occ_masked) - self.L2loss(f_clean_masked,f_neg_masked) + self.margin,0])

        score_clean = self.classifier(fc, labels)
        loss_clean = self.loss_cls(score_clean, labels)
        
        score_occ = self.classifier(fc_occ, labels)
        loss_occ = self.loss_cls(score_occ, labels)
        
        lamb = 10
        loss = 0.5 * loss_clean + 0.5 * loss_occ + 10 * triplet_loss
        
        _, pred_clean = torch.max(score_clean, dim=1)
        acc_clean = accuracy_score(pred_clean.cpu(), labels.cpu())
        acc_clean = torch.tensor(acc_clean)
        
        _, pred_occ = torch.max(score_occ, dim=1)
        acc_occ = accuracy_score(pred_occ.cpu(), labels.cpu())
        acc_occ = torch.tensor(acc_occ)
        
        return {'val_loss': loss, 'val_acc_clean':acc_clean, 'val_acc_occ':acc_occ}
        #return {'val_loss': loss, 'val_acc_occ':acc_occ}

    def validation_epoch_end(self, outputs):
        avgLoss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avgAcc_clean = torch.stack([x['val_acc_clean'] for x in outputs]).mean()
        avgAcc_occ = torch.stack([x['val_acc_occ'] for x in outputs]).mean()
        tensorboardLogs = {'val_loss': avgLoss, 'val_acc_clean':avgAcc_clean, 'val_acc_occ':avgAcc_occ}
        return {'val_loss': avgLoss, 'log': tensorboardLogs}



