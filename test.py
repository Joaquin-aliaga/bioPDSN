from lib.face import FaceVerificator
from lib.bioapi import FaceBio

import argparse
import torch
from pytorch_lightning import Trainer
from pathlib import Path
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params for face verification test')
    
    #data args
    #parser.add_argument("-dfPath","--dfPath",help="Path to dataframe",default=None,type=str)
    parser.add_argument("-test_database","--test_database",choices=["easy","hard","mask","nonmask"])
    #test args
    parser.add_argument("-b","--batch_size",help="batch size", default=1,type=int)
    parser.add_argument("-num_workers","--num_workers",help="num workers", default=4, type=int)
    parser.add_argument("-lr","--lr",help="Starting learning rate", default=1.0e-1,type=float)
    #parser.add_argument("-num_class","--num_class",default=403,help="Number of people (class)", type=int)
    parser.add_argument("-max_epochs","--max_epochs",help="Maximum epochs to train",default=10,type=int)

    #model args

    parser.add_argument("-model","--model",choices = ["PDSN","TRIPLET","ARCFACE","BIOAPI","RC-TRIPLET"],help="Which model weights use: [TrainDB]-[Loss]",default=None,type=str)
    parser.add_argument("-train_database","--train_database",choices=["RMFD","CASIA","RMFD-CASIA",None],help="Which trained weights to load")
    parser.add_argument("-backbone","--backbone",default="RESNET")
    parser.add_argument("-i", "--input_size", help="input size", default="3,112,112", type=str)
    parser.add_argument("-e", "--embedding_size", help="embedding size",default=512, type=int)
    parser.add_argument("-rw", "--resnet_weights", help="Path to resnet weights", default="./weights/model-r50-am-lfw/model,00",type=str)
    parser.add_argument("-post_process","--post_process",help="Whether use normalization after mtcnn (0=disable, 1=enable)",default=0,type=int)
    #parser.add_argument("-resnet_embedding","--resnet_embedding",choices=[0,1],default=0,help="Wheter use resnet embedding(1) or resnet last conv(0) as output")

    args = parser.parse_args()

    #not use resnet embedding, instead use resnet last conv layer as output.
    if args.train_database == "RMFD":
        args.num_class = 403
        if args.model == "PDSN":
            args.model_weights = "./checkpoints_pdsn_rmfd/epoch=19-val_acc_occ=0.98.ckpt"
        elif args.model == "TRIPLET":
            args.model_weights = "./checkpoints_triplet_rmfd/epoch=18-val_acc_occ=0.98.ckpt"
        else:
            print("Wrong choice of model with RMFD train database!")
            exit(1)
    elif args.train_database == "CASIA":
        args.num_class = 395
        if args.model == "PDSN":
            args.model_weights = "./checkpoints_pdsn_CASIA/epoch=19-val_acc_occ=0.51.ckpt"
        elif args.model == "TRIPLET":
            args.model_weights = "./checkpoints_triplet_CASIA/epoch=19-val_acc_occ=0.52.ckpt"
        else:
            print("Wrong choice of model with CASIA train database!")
            exit(1)
    elif args.train_database == "RMFD-CASIA":
        args.num_class = 395
        args.model_weights = "./checkpoints_triplet_RMFD-CASIA/epoch=19-val_acc_occ=0.53.ckpt"
    else:
        assert(args.model == "BIOAPI" or args.model =="ARCFACE")

    if(args.model == "ARCFACE"):
        args.backbone = args.model
 
    args.dfPath = "./lib/data/BioDBv3/{}_dataframe.pickle".format(args.test_database)

    #make test folder
    test_folder = './test'
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    
    #make model results folder
    if(args.model == "BIOAPI" or args.model == "ARCFACE"):
        model_folder = args.model
    else:
        model_folder = "{}-{}".format(args.train_database,args.model)
    
    results_folder = os.path.join(test_folder,model_folder)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    if(args.model == "BIOAPI"):
        model = FaceBio(args)
    else:
        model = FaceVerificator(args)
        
    model.to(args.device)

    #model.test() should return a DF with scores
    output = model.test()
    print("Test finished!")

    save_folder = test_folder+'/{}/'.format(args.model)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    dfName = save_folder+'{}_outputs.pickle'.format(args.test_database)
    print(f'saving Dataframe to: {dfName}')
    output.to_pickle(dfName)
    



