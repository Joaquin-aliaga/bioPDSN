import numpy as np
from easydict import EasyDict as edict
import mxnet as mx
import torch.nn as nn

#Resnet50 class (Arcface model)
class Resnet(nn.Module):
  def __init__(self,args):
    super(Resnet,self).__init__()
    self.imageShape = [int(x) for x in args.input_size.split(',')]
    self.vec = args.resnet_weights.split(',')
    self.batch_size = args.batch_size
    self.ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    #self.ctx = mx.cpu()
    #self.internals = None
    self.use_arcface = True if args.model == "ARCFACE" else False
    self.net = self.load_feature_model()

  def load_feature_model(self):
    assert len(self.vec)>1
    prefix = self.vec[0]
    epoch = int(self.vec[1])
    print('loading', prefix, epoch)
    net = edict()
    net.ctx = self.ctx
    net.sym, net.argParams, net.auxParams = mx.model.load_checkpoint(prefix, epoch)
    self.internals = net.sym.get_internals()
    if(self.use_arcface):
      #use resnet embeddings as output
      print("Using resnet's embedding")
      net.sym = self.internals['fc1_output']
    else:
      print("Using resnet's last convolutional layer")
      #use resnet last convolutional layer as output
      net.sym = self.internals['bn1_output']
    net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names = None)
    net.model.bind(data_shapes=[('data', (self.batch_size, self.imageShape[0], self.imageShape[1], self.imageShape[2]))])
    net.model.set_params(net.argParams, net.auxParams)
    
    return net
  
  def get_embedding(self,batch):

    if(len(batch.shape)>3): #batch with more than 1 element
      inputBlob = np.zeros( (self.batch_size, self.imageShape[0], self.imageShape[1], self.imageShape[2]) )
      idx = 0
      for img in batch:
        inputBlob[idx] = img
        idx+=1

    else: #batch with one element
      batch = batch.numpy()
      batch = np.expand_dims(batch,0) 
      inputBlob = batch
      
    data = mx.nd.array(inputBlob)
    db = mx.io.DataBatch(data=(data,))
    self.net.model.forward(db, is_train=False)
    features = self.net.model.get_outputs()[0].asnumpy()
    return features

  def forward(self,source,target):
    source_embedding = self.get_embedding(source)
    target_embedding = self.get_embedding(target)
    #return None,None,source,target because Biopdsn return this way
    return None,None,source_embedding,target_embedding
