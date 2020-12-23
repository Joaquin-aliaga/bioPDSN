import os

from lib.Biopdsn_triplet import BioPDSN
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch


if __name__ == '__main__':
    #data args
    parser = argparse.ArgumentParser(description='Params for bioPDSN train')
    #parser.add_argument("-dfPath","--dfPath",help="Path to dataframe",type=str)
    parser.add_argument("-train_database","--train_database",choices=['RMFD','CASIA'],type=str,help="Which Database use to train")

    #train args
    parser.add_argument("-b","--batch_size",help="batch size", default=32,type=int)
    parser.add_argument("-num_workers","--num_workers",help="num workers", default=4, type=int)
    parser.add_argument("-lr","--lr",help="Starting learning rate", default=1.0e-3,type=float)
    #parser.add_argument("-num_class","--num_class",help="Number of people (class)", type=int)
    parser.add_argument("-max_epochs","--max_epochs",help="Maximum epochs to train",default=10,type=int)
    parser.add_argument("--save_path","--save_path",help="Folder to save model checkpoints")

    #model args
    parser.add_argument("-i", "--input_size", help="input size", default="3,112,112", type=str)
    parser.add_argument("-e", "--embedding_size", help="embedding size",default=512, type=int)
    parser.add_argument("-rw", "--resnet_weights", help="Path to resnet weights", default="./weights/model-r50-am-lfw/model,00",type=str)
    
    args = parser.parse_args()

    if args.train_database == 'RMFD':
        args.dfPath = "./lib/data/dataframe_negatives.pickle"
        args.num_class = 403
    elif args.train_database == 'CASIA':
        args.dfPath = "./lib/data/CASIA_dataframe_negatives.pickle"
        args.num_class = 1003 #this number may change if you create CASIA_dataframe 
                            #the number of identities is prompted when you create it.
    else:
        print("Wrong train database")
        exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    biopdsn = BioPDSN(args).to(device)

    logger = TensorBoardLogger('triplet_{}_logs',name="triplet_{}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_triplet_{}/'.format(args.train_database),
        filename='{epoch}-{val_acc_occ:.2f}',
        save_weights_only=True,
        verbose=True,
        monitor='val_acc_occ',
        mode='max'
    )
    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                    auto_lr_find=True,
                    max_epochs=args.max_epochs,
                    checkpoint_callback=checkpoint_callback,
                    profiler=True,
                    logger = logger
                    )
    #find best learning rate
    trainer.tune(biopdsn)

    #train
    trainer.fit(biopdsn)
