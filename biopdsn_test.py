import os
import pandas as pd
import numpy as np
from lib.Biopdsn import BioPDSN
from lib.models.layer import cosine_sim
import argparse
import torch
import cv2

if __name__ == '__main__':
    #data args
    parser = argparse.ArgumentParser(description='Params for bioPDSN train')
    parser.add_argument("-dfPath","--dfPath",help="Path to dataframe",default=None,type=str)
    parser.add_argument("-test_folder","--test_folder",help="Path to test folder",default=None,type=str)

    #train args
    parser.add_argument("-b","--batch_size",help="batch size", default=32,type=int)
    parser.add_argument("-num_workers","--num_workers",help="num workers", default=4, type=int)
    parser.add_argument("-lr","--lr",help="Starting learning rate", default=1.0e-1,type=float)
    parser.add_argument("-num_class","--num_class",help="Number of people (class)", type=int)
    parser.add_argument("-max_epochs","--max_epochs",help="Maximum epochs to train",default=10,type=int)

    #model args
    parser.add_argument("-model_resume","--model_resume",help="Wheter use trained weights",default=True,type=bool)
    parser.add_argument("-model_weights","--model_weights",help="Path to model (trained) weights",default=None,type=str)
    parser.add_argument("-use_mtcnn","--use_mtcnn",help="Wheter use MTCNN to detect face",default="True",type=str)
    parser.add_argument("-i", "--input_size", help="input size", default="3,112,112", type=str)
    parser.add_argument("-e", "--embedding_size", help="embedding size",default=512, type=int)
    #parser.add_argument("-device", "--device", help="Which device use (cpu or gpu)", default='cpu', type=str)
    parser.add_argument("-rw", "--resnet_weights", help="Path to resnet weights", default="./weights/model-r50-am-lfw/model,00",type=str)
    parser.add_argument("-mtcnn_norm","--mtcnn_norm",help="Whether norm input after mtcnn",default=False,type=bool)
    parser.add_argument("-k","--keep_all",help="Wheter use all faces detected or just one with highest prob",default=False,type=bool)

    args = parser.parse_args()
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = BioPDSN(args)

    if args.model_resume:
        print("Loading model weights (trained)...")
        model.load_state_dict(torch.load(args.model_weights)['state_dict'], strict=False)
        print("Model weights loaded!")
    model = model.to(device)
    model.eval()
    root_folder_pos = args.test_folder+'/mascarillas_positivos/'
    root_folder_neg = args.test_folder+'/mascarillas_negativos/'
    print("Loading dataframes")
    df_pos = pd.read_csv(root_folder_pos+'pairs.csv')
    df_neg = pd.read_csv(root_folder_neg+'pairs.csv')
    print("Dataframes loaded!")
    row_pos = df_pos.sample().iloc[0]
    source_pos = cv2.imdecode(root_folder_pos + np.fromfile(row['ImgEnroll'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    target_pos = cv2.imdecode(root_folder_pos + np.fromfile(row['ImgQuery'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    f_clean_masked, f_occ_masked, fc_pos, fc_occ_pos, f_diff, mask = model(source_pos,target_pos)

    sim = cosine_sim(fc_pos,fc_occ_pos)
    print("Similitud positivos: ",sim)

    row_neg = df_neg.sample().iloc[0]
    source_neg = cv2.imdecode(root_folder_neg + np.fromfile(row['ImgEnroll'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    target_neg = cv2.imdecode(root_folder_neg + np.fromfile(row['ImgQuery'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    f_clean_masked, f_occ_masked, fc_neg, fc_occ_neg, f_diff, mask = model(source_neg,target_neg)

    sim = cosine_sim(fc_neg,fc_occ_neg)
    print("Similitud negativos: ",sim)


        

    


