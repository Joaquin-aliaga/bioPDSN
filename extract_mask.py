import os

from lib.Biopdsn import BioPDSN
import argparse

import torch

import sklearn.preprocessing

import numpy as np

if __name__ == '__main__':
    #data args
    parser = argparse.ArgumentParser(description='Params for bioPDSN train')
    parser.add_argument("-dfPath","--dfPath",help="Path to dataframe",type=str)

    #train args
    parser.add_argument("-num_class","--num_class",help="Number of people (class)", type=int)
    parser.add_argument("-b","--batch_size",help="batch size", default=32,type=int)
    parser.add_argument("-num_workers","--num_workers",help="num workers", default=4, type=int)
    parser.add_argument("-lr","--lr",help="Starting learning rate", default=1.0e-1,type=float)
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
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    biopdsn = BioPDSN(args).to(device)
    biopdsn.eval()

    biopdsn.prepare_data()

    train_dataloader = biopdsn.train_dataloader()

    with torch.no_grad():

        for batch_idx,batch in enumerate(train_dataloader):
            sources, targets, labels = batch['source'], batch['target'],batch['class']
            labels = labels.flatten()
            f_clean_masked, f_occ_masked, fc, fc_occ, f_diff, masks = biopdsn(sources,targets)
            print("Mask shape: ",masks.shape)

            masks_cpu = masks.to('cpu')
            masks_cpu = masks_cpu.view(-1,1)
            min_max_scaler = sklearn.preprocessing.MinMaxScaler()
            masks_cpu = min_max_scaler.fit_transform(masks_cpu)
            print("Mask shape after MinMax: ",masks_cpu.shape)
            print("First 7 elements: ",masks_cpu[0:8])
            mask_reshape = np.reshape(masks_cpu,masks.shape)
            print("Mask shape after reshape: ",mask_reshape.shape)
            print("First row elements of mask_reshape: ",mask_reshape[0,0,:,0])

            #OUT_SUM = OUT_SUM + mask_cpu.flatten()
            #count_p = count_p + 1
            break
        
        
        



