from pathlib import Path
import pandas as pd
from tqdm import tqdm
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

    for subject in tqdm(list(nonMaskPath.iterdir()), desc='no masked photos'):
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

    dfName = './merged_df.pickle'
    print(f'saving Dataframe to: {dfName}')
    merge.to_pickle(dfName)

    