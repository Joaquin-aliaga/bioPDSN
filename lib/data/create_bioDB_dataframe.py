"""
@author Joaquin Aliaga Gonzalez
@email joaliaga.g@gmail.com
@create date 2020-12-29 17:42:20
@modify date 2020-12-30 19:17:07
@desc [description]
"""
import pandas as pd

def concat_dataframes(root,path_pos,path_neg):
    path_pos = root+path_pos,
    path_neg = root+path_neg
    names = ['id','source','target']
    df_pos = pd.read_csv(path_pos+'pairs.csv',names=names)
    df_neg = pd.read_csv(path_neg+'pairs.csv',names=names)

    #create and add class for pos/neg cases
    pos_class = [1 for i in range(df_pos.shape[0])]
    neg_class = [0 for i in range(df_neg.shape[0])]

    df_pos['id_class'] = pos_class
    df_neg['id_class'] = neg_class

    # add root to paths
    df_pos.source = path_pos+df_pos.source
    df_pos.target = path_pos+df_pos.target

    df_neg.source = path_neg+df_neg.source
    df_neg.target = path_neg+df_neg.target

    concat = pd.concat([df_pos,df_neg])
    #shuffle dataframe
    concat = concat.sample(frac=1)

    return concat

if __name__ == "__main__":
    
    root = 'BioDBv3/'
    
    pos_easy = 'positivos_faciles/'
    neg_easy = 'negativos_faciles/'

    print("Creating easy examples dataframe")
    dataframe = concat_dataframes(root,pos_easy,neg_easy)
    name = './'+root+'easy_dataframe.pickle'
    print(f'saving Dataframe to: {name}')
    dataframe.to_pickle(name)

    pos_hard = 'positivos_dificiles/'
    neg_hard = 'negativos_dificiles/'

    print("Creating hard examples dataframe")
    dataframe = concat_dataframes(root,pos_hard,neg_hard)
    name = './'+root+'hard_dataframe.pickle'
    print(f'saving Dataframe to: {name}')
    dataframe.to_pickle(name)
    
    pos_mask = 'mascarillas/mascarillas_positivos/'
    neg_mask = 'mascarillas/mascarillas_negativos/'

    print("Creating masked dataframe")
    dataframe = concat_dataframes(root,pos_mask,neg_mask)
    name = './'+root+'mask_dataframe.pickle'
    print(f'saving Dataframe to: {name}')
    dataframe.to_pickle(name)
    
    pos_nonmask = 'sin_mascarillas/positivos_dificiles/'
    neg_nonmask = 'sin_mascarillas/negativos_dificiles/'

    print("Creating non-masked dataframe")
    dataframe = concat_dataframes(root,pos_nonmask,neg_nonmask)
    name = './'+root+'nonmask_dataframe.pickle'
    print(f'saving Dataframe to: {name}')
    dataframe.to_pickle(name)

    print("Dataframes creation finished!.")
