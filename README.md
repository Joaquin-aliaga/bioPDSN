# bioPDSN
Face verification system for people wearing masks, based on Pairwise Differential Siamese Network [paper](https://arxiv.org/abs/1908.06290)

## System
* Ubuntu 18
* Python 3.6
* Cuda 10.1

## Clone repo
`git clone https://github.com/Joaquin-aliaga/bioPDSN.git`

## Install dependencies
* `pip install -r requirements.txt` (not recommended, you may have memory issues depending on your system)
* A better way is to install libraries one-by-one following requirements.txt

## Prepare Data
1. Download and unzip databases
    * Download RMFD dataset from [here](https://drive.google.com/file/d/1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp/view?usp=sharing) and put it in lib/data
    * `cd lib/data`
    * `unzip RMFD.zip`
    * Download CASIA-webface-112x112 from
    * run `create_casiawebface_112x112.py`
    * `unzip CASIA-webface.zip`
2. Create dataframes (inside lib/data)
    * `python create_dataframe.py --use_database "RMFD"`
    * `python create_dataframe.py --use_database "CASIA"`

## Train
1. Download LResNet50E-IR pretrained from [insightface Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) and put it in ./weights
2. run `python biopdsn_train.py -num_class 403 -dfPath "./lib/data/{RMFD or CASIA}_dataframe.pickle" -rw "./weights/model-r50-am-lfw/model,00" --batch_size 64`

### Usefull guides
* [Create VM instance Google Cloud Platform](https://cloud.google.com/ai-platform/deep-learning-vm/docs/pytorch_start_instance)
* Create requirements.txt -> `pip install pipreqs` -> `pipreqs .`
* wget from Drive (large)file: `wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt`
* curl from Dropbox: `curl -L <dropbox-link>?dl=1 > <file-name>`

### Acknowledgments
* The Real Masked Face Dataset (RMFD) is owned by [https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)
* The Casia-webface-masked database was created using a modified version of [https://github.com/Amoswish/wear_mask_to_face](https://github.com/Amoswish/wear_mask_to_face) 