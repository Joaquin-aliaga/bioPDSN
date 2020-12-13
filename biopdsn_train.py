#import torch
#from easydict import EasyDict as edict
from lib.Biopdsn import BioPDSN
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import torch


if __name__ == '__main__':
    #data args
    parser = argparse.ArgumentParser(description='Params for bioPDSN train')
    parser.add_argument("-dfPath","--dfPath",help="Path to dataframe",type=str)

    #train args
    parser.add_argument("-b","--batch_size",help="batch size", default=32,type=int)
    parser.add_argument("-num_workers","--num_workers",help="num workers", default=4, type=int)
    parser.add_argument("-lr","--lr",help="Starting learning rate", default=1.0e-1,type=float)
    parser.add_argument("-num_class","--num_class",help="Number of classes", type=int)
    parser.add_argument("-max_epochs","--max_epochs",help="Maximum epochs to train",default=10,type=int)

    #model args
    parser.add_argument("-use_mtcnn","--use_mtcnn",help="Wheter use MTCNN to detect face",default="False",type=str)
    parser.add_argument("-i", "--input_size", help="input size", default="3,112,112", type=str)
    parser.add_argument("-e", "--embedding_size", help="embedding size",default=512, type=int)
    #parser.add_argument("-device", "--device", help="Which device use (cpu or gpu)", default='cpu', type=str)
    parser.add_argument("-rw", "--resnet_weights", help="Path to resnet weights", default="./weights/model-r50-am-lfw/model,00",type=str)
    parser.add_argument("-mtcnn_norm","--mtcnn_norm",help="Whether norm input after mtcnn",default=True,type=bool)
    parser.add_argument("-k","--keep_all",help="Wheter use all faces detected or just one with highest prob",default=False,type=bool)

    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    biopdsn = BioPDSN(args).to(device)
    
    checkpoint_callback = ModelCheckpoint(
        filepath='./checkpoints/weights.ckpt',
        save_weights_only=True,
        verbose=True,
        monitor='val_acc_occ',
        mode='max'
    )
    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                      max_epochs=args.max_epochs,
                      checkpoint_callback=checkpoint_callback,
                      profiler=True)
    trainer.fit(biopdsn)