#import torch
#from easydict import EasyDict as edict
from lib.Biopdsn import BioPDSN
import argparse
from facenet_pytorch import MTCNN


if __name__ == '__main__':
    #must be given
    parser = argparse.ArgumentParser(description='for arcface test')
    parser.add_argument("-p","--dfPath",help="Path to dataframe",type=str)
    parser.add_argument("-num_class","--num_class",help="Number of classes", type=int)

    #default
    parser.add_argument("-i", "--input_size", help="input size", default="3,112,112", type=str)
    parser.add_argument("-b","--batch_size",help="batch size", default=32,type=int)
    parser.add_argument("-workers","--num_workers",help="num workers", default=4, type=int)
    parser.add_argument("-e", "--embedding_size", help="embedding size",default=512, type=int)
    parser.add_argument("-u", "--use_mobilefacenet", help="Wheter use mobilefacenet ", default=False, type=bool)
    parser.add_argument('-d','--net_depth',help='how many layers [50,100,152]',default=50, type=int)
    parser.add_argument("-n", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-threshold", "--threshold", help="Threshold to use in verification", default=0.5, type=float)
    parser.add_argument("-drop", "--drop_ratio", help="Drop ratio", default=0.6, type=float)
    parser.add_argument("-lr","--lr",help="Starting learning rate", default=0.02,type=float)
    parser.add_argument("-device", "--device", help="Which device use (cpu or gpu)", default='cpu', type=str)
    parser.add_argument("-rw", "--resnet_weights", help="Path to resnet weights", default="./weights/model-r50-am-lfw/model,00",type=str)
    parser.add_argument("-transform","--transform", help="Input transform",default=False,type=bool)
    parser.add_argument("-images","--images_path", help="Path to images",default=None,type=str)
    parser.add_argument("-mtcnn_norm","--mtcnn_norm",help="Whether norm input after mtcnn",default=True,type=bool)
    parser.add_argument("-k","--keep_all",help="Wheter use all faces detected or just one with highest prob",default=False,type=bool)

    args = parser.parse_args()

    biopdsn = BioPDSN(args).to(args.device)
    
    #print(biopdsn.get_parameters())
    for name, param in biopdsn.named_parameters():
        if param.requires_grad == True:
            if not 'mtcnn' in name:
                #params_to_update.append(param)
                print("Update \t", name)
    '''
    Freeze parameters
    # we want to freeze the fc2 layer this time: only train fc1 and fc3
    net.fc2.weight.requires_grad = False
    net.fc2.bias.requires_grad = False

    # passing only those parameters that explicitly requires grad
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

    '''
