from models.arcface import  Backbone, Arcface, l2_norm
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
#from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
#plt.switch_backend('agg')
#from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
#import bcolz

class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        if not inference:
            
            """ self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = Arcface(embedding_size=conf.embedding_size, classnum = self.class_num).to(conf.device)

            print(" Two model heads generated")

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            self.optimizer = optim.SGD([
                {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                {'params': paras_only_bn}
                ], lr = conf.lr, momentum = conf.momentum)
            print(self.optimizer)
            print("Optimizers generated")
            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)//10
            self.save_every = len(self.loader)//5
            """
             #self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(self.loader.dataset.root.parent)
        else:
            self.threshold = conf.threshold

    """ def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
     """
    def load_state(self, conf, fixed_str, from_save_folder = False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path

        self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))

        if not model_only:
            self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))

    """ def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
 """
    def infer(self,conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''

        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device()).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device)).unsqueeze(0))
        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff,2), dim=1)

        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 #if no match, set idx to -1
        return min_idx, minimum
#   

