from PIL import Image
import cv2
import os
import mxnet as mx
from tqdm import tqdm
import argparse

'''
For train dataset, insightface provide a mxnet .rec file, just install a mxnet-cpu for extract images
'''

def load_mx_rec(rec_path,save_folder):
  save_path = os.path.join(rec_path, save_folder)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'), os.path.join(rec_path, 'train.rec'), 'r')
  img_info = imgrec.read_idx(0)
  header,_ = mx.recordio.unpack(img_info)
  max_idx = int(header.label[0])
  for idx in tqdm(range(1,max_idx)):
    img_info = imgrec.read_idx(idx)
    header, img = mx.recordio.unpack_img(img_info)
    label = int(header.label)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    label_path = os.path.join(save_path, str(label).zfill(6))
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    img.save(os.path.join(label_path, str(idx).zfill(8) + '.jpg'), quality=95)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create casiawebface 112x112 dataset from .rec file')
    parser.add_argument("-r", "--rec_path", help="Path to .rec folder, .idx file must be in the same folder",default='./',type=str)
    parser.add_argument('-s','--save_folder',help='Folder to save images',default="casia_webface_112x112", type=str)
    args = parser.parse_args()

    load_mx_rec(args.rec_path,args.save_folder)