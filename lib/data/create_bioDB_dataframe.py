"""
@author Joaquin Aliaga Gonzalez
@email joaliaga.g@gmail.com
@create date 2020-12-29 17:42:20
@modify date 2020-12-29 18:13:18
@desc [description]
"""
import pandas as pd
import os
from tqdm import tqdm
import cv2

names = ['id','source','target']

root = '/BioDBv3'
root_pos = '/positivos'
root_neg = '/negativos'
root_mask_pos = '/mascarillas/mascarillas_positivos'
root_mask_neg = '/mascarillas/mascarillas_negativos'
root_nonmask_pos = '/sin_mascarillas/positivos_dificiles'
root_nonmask_neg = '/sin_mascarillas/negativos_dificiles'

casos = ['_faciles/,_dificiles/']

# positive and negative cases
for caso in casos:
    print("Creando dataframe para casos: {}".format(caso))
    pre_path_pos = root+root_pos+caso
    pre_path_neg = root+root_neg+caso

    df_pos = pd.read_csv(pre_path_pos+'pairs.csv')
    df_neg = pd.read_csv(pre_path_neg+'pairs.csv')

    #create and add class for pos/neg cases
    pos_class = [1 for i in range(df_pos.shape[0])]
    neg_class = [0 for i in range(df_neg.shape[0])]

    df_pos['id_class'] = pos_class
    df_neg['id_class'] = neg_class

    # add root to paths
    df_pos.source = pre_path_pos+df_pos.source
    df_pos.target = pre_path_pos+df_pos.target

    df_neg.source = pre_path_neg+df_neg.source
    df_neg.target = pre_path_neg+df_neg.target

    concat = pd.concat([df_pos,df_neg])
    #shuffle dataframe
    concat = concat.sample(frac=1)

# mask cases
for caso in casos:
    print("Creando dataframe para casos: {}".format(caso))
    pre_path_pos = root+root_pos+caso
    pre_path_neg = root+root_neg+caso

    df_pos = pd.read_csv(pre_path_pos+'pairs.csv')
    df_neg = pd.read_csv(pre_path_neg+'pairs.csv')

    #create and add class for pos/neg cases
    pos_class = [1 for i in range(df_pos.shape[0])]
    neg_class = [0 for i in range(df_neg.shape[0])]

    df_pos['id_class'] = pos_class
    df_neg['id_class'] = neg_class

    # add root to paths
    df_pos.source = pre_path_pos+df_pos.source
    df_pos.target = pre_path_pos+df_pos.target

    df_neg.source = pre_path_neg+df_neg.source
    df_neg.target = pre_path_neg+df_neg.target

    concat = pd.concat([df_pos,df_neg])
    #shuffle dataframe
    concat = concat.sample(frac=1)


