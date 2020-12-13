# bioPDSN
Face verification system using Pairwise Differential Siamese Network [paper](https://arxiv.org/abs/1908.06290)

##Requirements
* Python 3.x
* Pytorch
* facenet-pytorch
* mxnet-cu101
* pytorch-lightning


## Train
0. `git clone https://github.com/Joaquin-aliaga/bioPDSN.git`
1.   Download RMFD dataset from [here](https://drive.google.com/file/d/1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp/view?usp=sharing) and put it in lib/data
2. `cd lib/data && unzip RMFD.zip`
3. (inside lib/data) `python create_rmfd_dataframe.py`
4. `cd ../..`
5. Download LResNet50E-IR pretrained from [insightface Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) and put it in ./weights
6. python biopdsn_train.py -num_class 403 -use_mtcnn "False" -dfPath "./lib/data/dataframe.pickle" -rw "./weights/model-r50-am-lfw/model,00" -b 64