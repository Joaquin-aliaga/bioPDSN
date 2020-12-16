import os
import pandas as pd
import numpy as np
from lib.Biopdsn import BioPDSN
from lib.models.layer import cosine_sim, MarginCosineProduct
import argparse
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
import cv2
from facenet_pytorch import MTCNN
from PIL import Image


if __name__ == '__main__':
    #data args
    parser = argparse.ArgumentParser(description='Params for bioPDSN train')
    parser.add_argument("-dfPath","--dfPath",help="Path to dataframe",default=None,type=str)
    parser.add_argument("-test_folder","--test_folder",help="Path to test folder",default=None,type=str)

    #train args
    parser.add_argument("-b","--batch_size",help="batch size", default=1,type=int)
    parser.add_argument("-num_workers","--num_workers",help="num workers", default=4, type=int)
    parser.add_argument("-lr","--lr",help="Starting learning rate", default=1.0e-1,type=float)
    parser.add_argument("-num_class","--num_class",help="Number of people (class)", type=int)
    parser.add_argument("-max_epochs","--max_epochs",help="Maximum epochs to train",default=10,type=int)

    #model args
    parser.add_argument("-model_resume","--model_resume",help="Wheter use trained weights",default=True,type=bool)
    parser.add_argument("-model_weights","--model_weights",help="Path to model (trained) weights",default=None,type=str)
    parser.add_argument("-use_mtcnn","--use_mtcnn",help="Wheter use MTCNN to detect face",default="False",type=str)
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
    
    imageShape = [int(x) for x in args.input_size.split(',')]

    transformations = Compose([
            ToPILImage(),
            ToTensor(), # [0, 1]
        ])
        
    mtcnn = MTCNN(image_size=imageShape[1], min_face_size=20, 
                            device = device, post_process=args.mtcnn_norm,
                            keep_all=False,select_largest=True)

    model = BioPDSN(args)

    if args.model_resume:
        print("Loading model weights (trained)...")
        model.load_state_dict(torch.load(args.model_weights)['state_dict'], strict=False)
        print("Model weights loaded!")
    model = model.to(device)
    model.eval()

    pd_names = ['id','ImgEnroll','ImgQuery']
    root_folder_pos = args.test_folder+'/mascarillas_positivos/'
    root_folder_neg = args.test_folder+'/mascarillas_negativos/'
    print("Loading dataframes")
    df_pos = pd.read_csv(root_folder_pos+'pairs.csv',names=pd_names)
    df_neg = pd.read_csv(root_folder_neg+'pairs.csv',names=pd_names)
    print("Dataframes loaded!")
    
    row_pos = df_pos.sample().iloc[0]
    #source_pos = Image.open(root_folder_pos+row_pos['ImgEnroll'])
    #target_pos = Image.open(root_folder_pos+row_pos['ImgQuery'])
    source_pos = cv2.imdecode(np.fromfile(root_folder_pos + row_pos['ImgEnroll'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    target_pos = cv2.imdecode(np.fromfile(root_folder_pos + row['ImgQuery'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    source_pos = transformations(source_pos)
    target_pos = transformations(target_pos)

    source_pos = mtcnn(source_pos)
    target_pos = mtcnn(target_pos)
    print("Source shape after mtcnn: ",source_pos.shape)

    f_clean_masked, f_occ_masked, fc_pos, fc_occ_pos, f_diff, mask = model(source_pos,target_pos)

    print("fc pos shape:" ,fc_pos.shape)
    print("fc pos occ shape:" ,fc_occ_pos.shape)
    sim = cosine_sim(fc_pos,fc_occ_pos,dim=1)
    print("Similitud positivos: ",sim)
    '''

    row_neg = df_neg.sample().iloc[0]
    source_neg = Image.open(root_folder_neg+row_neg['ImgEnroll'])
    target_neg = Image.open(root_folder_neg+row_neg['ImgQuery'])

    source_neg = mtcnn(source_neg)
    target_neg = mtcnn(target_neg)

    f_clean_masked, f_occ_masked, fc_neg, fc_occ_neg, f_diff, mask = model(source_neg,target_neg)

    sim_neg = cosine_sim(fc_neg,fc_occ_neg,dim=0)
    print("Similitud negativos: ",sim_neg)
    '''


        

    


