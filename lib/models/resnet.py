import numpy as np
from easydict import EasyDict as edict
import mxnet as mx

#Resnet50 class (Arcface model)
class Resnet():
  def __init__(self,args):
    self.imageShape = [int(x) for x in args.input_size.split(',')]
    self.vec = args.resnet_weights.split(',')
    self.batch_size = args.batch_size
    self.ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    #self.ctx = mx.cpu()
    #self.internals = None
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
    #net.sym = self.internals['fc1_output']
    net.sym = self.internals['bn1_output']
    net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names = None)
    net.model.bind(data_shapes=[('data', (self.batch_size, self.imageShape[0], self.imageShape[1], self.imageShape[2]))])
    net.model.set_params(net.argParams, net.auxParams)
    
    return net
  
  def get_features(self,batch):
    '''
    batch is a list of torch.tensor items (aligned faces croped using mtcnn)
    '''
    inputBlob_sources = np.zeros( (self.batch_size, self.imageShape[0], self.imageShape[1], self.imageShape[2]) )
    idx = 0
    
    for img in batch:
      inputBlob[idx] = img
      idx+=1
    
    data = mx.nd.array(inputBlob)
    db = mx.io.DataBatch(data=(data,))
    self.net.model.forward(db, is_train=False)
    features = self.net.model.get_outputs()[0].asnumpy()
    return features