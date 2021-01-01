import os

from lib.Biopdsn import BioPDSN
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch

#CUDA_LAUNCH_BLOCKING=1

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

    #test args
    parser.add_argument("-t,--test_path",default=None, type=str,help="Path to test dataframe")

    #model args
    parser.add_argument("-i", "--input_size", help="input size", default="3,112,112", type=str)
    parser.add_argument("-e", "--embedding_size", help="embedding size",default=512, type=int)
    parser.add_argument("-rw", "--resnet_weights", help="Path to resnet weights", default="./weights/model-r50-am-lfw/model,00",type=str)
    
    args = parser.parse_args()

    if args.train_database == 'RMFD':
        args.dfPath = "./lib/data/dataframe.pickle"
        args.num_class = 403
    elif args.train_database == 'CASIA':
        args.dfPath = "./lib/data/CASIA_dataframe.pickle"
        args.num_class = 395 #this number may change if you create a new CASIA_dataframe. the number of identities is prompted when dataframe is created
    else:
        print("Wrong train database")
        exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #args.device = device
    biopdsn = BioPDSN(args).to(device)

    logger = TensorBoardLogger('pdsn_{}_logs'.format(args.train_database),name="pdsn_{}".format(args.train_database))
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_pdsn_{}/'.format(args.train_database),
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
                    profiler=True
                    )
    #find best starting lr
    trainer.tune(biopdsn)

    #train
    trainer.fit(biopdsn)
