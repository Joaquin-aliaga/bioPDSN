# bioPDSN
Face verification system for people wearing masks, based on Pairwise Differential Siamese Network [paper](https://arxiv.org/abs/1908.06290)

## System
* Ubuntu 18
* Python 3.6
* Cuda 10.1

## Clone repo
`git clone https://github.com/Joaquin-aliaga/bioPDSN.git`

- [Install dependencies](#install-dependencies)
- [Train Triplet Loss](#train-triplet-loss)

## Install dependencies
* `pip install -r requirements.txt` (not recommended, you may have memory issues depending on your system)
* A better way is to install libraries one-by-one following requirements.txt

## Prepare Data
1. Download and unzip databases
    * Download RMFD dataset from [here](https://drive.google.com/file/d/1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp/view?usp=sharing) and put it in lib/data
    * `cd lib/data`
    * `unzip RMFD.zip`
    * Download CASIA-webface-112x112 from [here](https://drive.google.com/file/d/1Pfn90QHx51gNlK1a6zzXCmfmNOetlVYy/view?usp=sharing)
    * `mkdir CASIA-webface && cd CASIA-webface`
    * `unzip CASIA-webface.zip`
2. Create dataframes (inside lib/data)
    * `python create_dataframe.py --use_database "RMFD"`
    * `python create_dataframe.py --use_database "CASIA"`

## Prepare pretrained model
1. Download LResNet50E-IR pretrained from [insightface Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) and put it in ./weights

## Train PDSN Loss
1. run `python biopdsn_train.py --train_database {"CASIA" or "RMFD"}`

## Train Contrastive Loss (not implemented yet)
1. run `python biopdsn_train_contrastive.py --train_database {"CASIA" or "RMFD"}`

## Train Triplet Loss
1. run `python biopdsn_train_triplet.py --train_database {"CASIA" or "RMFD"}`

## Test
1. First you need to put your images inside /lib/data/your-images
2. Second you need to create a dataframe that contains 'source' (source_path), 'target' (target_path) and 'class' (1/positive 0/negative pair) and save that dataframe inside lib/data/<some_name>_dataframe.pickle
3. run `python test.py --test_database "<some_name>" --model ["RMFD-PDSN" or "RMFD-TRIPLET" or "CASIA-PDSN" or "CASIA-TRIPLET"] -b 64 -num_workers 8`

### Usefull guides
* [Create VM instance Google Cloud Platform](https://cloud.google.com/ai-platform/deep-learning-vm/docs/pytorch_start_instance)
* Create requirements.txt -> `pip install pipreqs` -> `pipreqs .`
* wget from Drive (large)file: `wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt`
* curl from Dropbox: `curl -L <dropbox-link>?dl=1 > <file-name>`

### Acknowledgments
* The Real Masked Face Dataset (RMFD) is owned by [https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)
* The Casia-webface-masked database was created using a modified version of [https://github.com/Amoswish/wear_mask_to_face](https://github.com/Amoswish/wear_mask_to_face) 