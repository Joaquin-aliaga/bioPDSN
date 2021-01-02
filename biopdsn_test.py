from lib.face import FaceVerificator

import argparse
import torch
from pytorch_lightning import Trainer
from pathlib import Path

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

    parser.add_argument("-model","--model",choices = ["RMFD-PDSN","RMFD-TRIPLET","CASIA-PDSN","CASIA-TRIPLET"],help="Which model weights use: [TrainDB]-[Loss]",default=None,type=str)
    parser.add_argument("-i", "--input_size", help="input size", default="3,112,112", type=str)
    parser.add_argument("-e", "--embedding_size", help="embedding size",default=512, type=int)
    parser.add_argument("-rw", "--resnet_weights", help="Path to resnet weights", default="./weights/model-r50-am-lfw/model,00",type=str)
    parser.add_argument("-post_process","--post_process",help="Whether use normalization after mtcnn (0=disable, 1=enable)",default=0,type=int)
    
    args = parser.parse_args()

    if args.model == "RMFD-PDSN":
        args.model_weights = "./checkpoints_pdsn_rmfd/epoch=19-val_acc_occ=0.98.ckpt"
        args.num_class = 403
    elif args.model == "RMFD-TRIPLET":
        args.model_weights = "./checkpoints_triplet_rmfd/epoch=18-val_acc_occ=0.98.ckpt"
        args.num_class = 403
    elif args.model == "CASIA-PDSN":
        args.model_weights = "./checkpoints_pdsn_CASIA/epoch=19-val_acc_occ=0.51.ckpt"
        args.num_class = 395
    elif args.model == "CASIA-TRIPLET":
        args.model_weights = "./checkpoints_triplet_CASIA/epoch=19-val_acc_occ=0.52.ckpt"
        args.num_class = 395
    
    args.dfPath = "./lib/data/BioDBv3/{}_dataframe.pickle".format(args.test_database)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model = FaceVerificator(args).to(args.device)

    #model.test() should return a DF with scores
    output = model.test()
    print("Test finished!")

    print("Output type: ",type(output))
    print("Output shape: ",output.shape)

    dfName = './test/{}_outputs.pickle'.format(args.test_database)
    print(f'saving Dataframe to: {dfName}')
    output.to_pickle(dfName)
    



