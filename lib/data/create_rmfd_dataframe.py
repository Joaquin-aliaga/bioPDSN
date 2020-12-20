from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
'''
from google_drive_downloader import GoogleDriveDownloader as gdd
# download dataset from link provided by
# https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
datasetPath = Path('covid-mask-detector/data/mask.zip')
gdd.download_file_from_google_drive(file_id='1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp',
                                    dest_path=str(datasetPath),
                                    unzip=True)
# delete zip file
datasetPath.unlink()
'''

def create_negatives_column(df):
    negatives = pd.DataFrame()
    class_list = df.id_class.unique().tolist()
    for i in tqdm(range(df.shape[0]),desc="Creating negatives column"):
        row = df.iloc[i]
        while True:
            #select a random class from class_list
            negative_class = np.random.choice(class_list)

            #check the random class is different from real class
            if (negative_class != row.id_class):
                neg_df = df[df.id_class == negative_class]
                #select a random row
                row_neg = neg_df.sample().iloc[0]
                negatives = negatives.append({
                    'id_negative_class' : row.id_class,
                    'id_real_class' : row_neg.id_class,
                    'negative' : row_neg.target
                },ignore_index=True)
                break
    return negatives


if __name__ == '__main__':
    datasetPath = Path('./self-built-masked-face-recognition-dataset')
    maskPath = datasetPath/'AFDB_masked_face_dataset'
    nonMaskPath = datasetPath/'AFDB_face_dataset'
    maskDF = pd.DataFrame()
    nonMaskDF = pd.DataFrame()

    for subject in tqdm(list(maskPath.iterdir()), desc='mask photos'):
        id = subject.stem
        for imgPath in subject.iterdir():
            maskDF = maskDF.append({
                'target': str(imgPath),
                'id_name': str(id)
            }, ignore_index=True)

    for subject in tqdm(list(nonMaskPath.iterdir()), desc='non masked photos'):
        id = subject.stem
        for imgPath in subject.iterdir():
            nonMaskDF = nonMaskDF.append({
                'source': str(imgPath),
                'id_name': str(id)
            }, ignore_index=True)
    
    merge = pd.merge(nonMaskDF,maskDF,on='id_name')

    #adding class numbers to id_names
    merge['id_name'] = merge['id_name'].astype('category')
    merge['id_class'] = merge['id_name'].cat.codes

    dfName = './dataframe.pickle'
    print(f'saving Dataframe to: {dfName}')
    merge.to_pickle(dfName)
    
    negatives = create_negatives_column(merge)

    dfName = './negatives.pickle'
    print(f'saving Dataframe to: {dfName}')
    negatives.to_pickle(dfName)

    merge['negative'] = negatives.negative

    dfName = './dataframe_negatives.pickle'
    print(f'saving Dataframe to: {dfName}')
    merge.to_pickle(dfName)



    