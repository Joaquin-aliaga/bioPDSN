from lib.models.resnet import Resnet
from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
'''
args example
args = {
    "batch_size" : 1,
    "image_size" : '3,112,112',
    "resize" : (910,1240),
    "model_path" : "./model/model-r50-am-lfw/model,00",
    "mtcnn_norm": True,
    "keep_all": False
} 
'''

class BioPDSN(nn.Module):
    def __init__(self,args):
        super(BioPDSN,self).__init__()
        
        self.resize = args.resize
        self.imageShape = [int(x) for x in args.image_size.split(',')]
        self.features_shape = 512
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.mtcnn = MTCNN(image_size=self.imageShape[1], min_face_size=80, device = self.device, post_process=args.mtcnn_norm,keep_all=args.keep_all)

        self.resnet = Resnet(args)

        self.sia = nn.Sequential(
            #nn.BatchNorm2d(filter_list[4]),
            nn.Conv2d(self.features_shape, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(self.features_shape),
            nn.BatchNorm2d(self.features_shape),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.features_shape * 7 * 6),
            #nn.Dropout(p=0),
            nn.Linear(self.features_shape * 7 * 6, 512),
            nn.BatchNorm1d(512),
        )

        #cuanto exista el mask generator
        #self.mask = MaskGen()
    
    def get_faces(self,batch):
        if (type(batch) == list):
            batch = [img.resize(self.resize) for img in batch]
        return self.mtcnn(batch)

    
    def get_features(self,source,target):
        batch = [source,target]
        faces = self.get_faces(batch)
        features = self.resnet.get_features(faces) #type(features) = numpy ndarray

        return features

    def forward(self,source,target):
        f1,f2 = self.get_features(source,target)

        # Begin Siamese branch
        f_diff = torch.add(f1, -1.0, f2)
        f_diff = torch.abs(f_diff)
        out = self.sia(f_diff)
        # End Siamese branch

        f1_masked = f1 * out
        f2_masked = f2 * out

        fc1 = f1_masked.view(f1_masked.size(0), -1) #256*(512*7*6)
        fc2 = f2_masked.view(f2_masked.size(0), -1)
        fc1 = self.fc(fc1)
        fc2 = self.fc(fc2)

        return f1_masked, f2_masked, fc1, fc2, f_diff, out

