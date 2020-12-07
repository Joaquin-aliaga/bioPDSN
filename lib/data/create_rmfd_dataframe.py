import cv2
import 

datasetPath = Path('dataset/self-built-masked-face-recognition-dataset')
maskPath = datasetPath/'AFDB_masked_face_dataset'
nonMaskPath = datasetPath/'AFDB_face_dataset'
maskDF = pd.DataFrame()

for subject in tqdm(list(nonMaskPath.iterdir()), desc='non mask photos'):
    for imgPath in subject.iterdir():
        image = cv2.imread(str(imgPath))
        maskDF = maskDF.append({
            'image': image,
            'mask': 0
        }, ignore_index=True)

for subject in tqdm(list(maskPath.iterdir()), desc='mask photos'):
    for imgPath in subject.iterdir():
        image = cv2.imread(str(imgPath))
        maskDF = maskDF.append({
            'image': image,
            'mask': 1
        }, ignore_index=True)

maskDF.to_pickle('data/mask_df.pickle')