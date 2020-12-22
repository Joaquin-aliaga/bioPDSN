import os

from lib.Biopdsn import BioPDSN
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch


if __name__ == '__main__':
    wd = os.getcwd()
    print("Wd: ",wd)
    #data args
    parser = argparse.ArgumentParser(description='Params for bioPDSN train')
    parser.add_argument("-dfPath","--dfPath",help="Path to dataframe",type=str)

    #train args
    parser.add_argument("-b","--batch_size",help="batch size", default=32,type=int)
    parser.add_argument("-num_workers","--num_workers",help="num workers", default=4, type=int)
    parser.add_argument("-lr","--lr",help="Starting learning rate", default=1.0e-3,type=float)
    parser.add_argument("-num_class","--num_class",help="Number of people (class)", type=int)
    parser.add_argument("-max_epochs","--max_epochs",help="Maximum epochs to train",default=10,type=int)

    #model args
    #parser.add_argument("-use_mtcnn","--use_mtcnn",help="Wheter use MTCNN to detect face",default="False",type=str)
    parser.add_argument("-i", "--input_size", help="input size", default="3,112,112", type=str)
    parser.add_argument("-e", "--embedding_size", help="embedding size",default=512, type=int)
    #parser.add_argument("-device", "--device", help="Which device use (cpu or gpu)", default='cpu', type=str)
    parser.add_argument("-rw", "--resnet_weights", help="Path to resnet weights", default="./weights/model-r50-am-lfw/model,00",type=str)
    #parser.add_argument("-mtcnn_norm","--mtcnn_norm",help="Whether norm input after mtcnn",default=True,type=bool)
    #parser.add_argument("-k","--keep_all",help="Wheter use all faces detected or just one with highest prob",default=False,type=bool)

    args = parser.parse_args()
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    biopdsn = BioPDSN(args).to(device)

    #logger = TensorBoardLogger('tb_logs',name="pdsn_contrastive_rmfd")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_contrastive_rmfd/',
        filename='{epoch}-{val_acc_occ:.2f}',
        save_weights_only=True,
        #save_top_k=1,
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