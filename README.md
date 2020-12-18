# bioPDSN
Face verification system using Pairwise Differential Siamese Network [paper](https://arxiv.org/abs/1908.06290)

## System
* Ubuntu 18
* Python 3.6
* Cuda 10.1


## Install dependencies
`pip install -r requirements.txt`

## Train
0. `git clone https://github.com/Joaquin-aliaga/bioPDSN.git`
1.   Download RMFD dataset from [here](https://drive.google.com/file/d/1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp/view?usp=sharing) and put it in lib/data
2. `cd lib/data && unzip RMFD.zip`
3. (inside lib/data) `python create_rmfd_dataframe.py`
4. `cd ../..`
5. Download LResNet50E-IR pretrained from [insightface Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) and put it in ./weights
6. run `python biopdsn_train.py -num_class 403 -dfPath "./lib/data/dataframe.pickle" -rw "./weights/model-r50-am-lfw/model,00" --batch_size 64`

### Usefull guides
* [Install Cuda10.1](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130)
* Create requirements.txt -> `pip install pipreqs` -> `pipreqs .`
* wget from Drive (large)file: `wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt`
* wget from Dropbox: `curl -L <dropbox-link>?dl=1 > <file-name>`
* random.