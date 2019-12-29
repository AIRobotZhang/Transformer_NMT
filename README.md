# Transformer for Neural Machine Translation

## Requirements
* Python==3.7.4
* pytorch==1.3.1
* torchtext==0.3.1
* numpy
* tqdm
* jieba
* nltk

## Input data format
Sample dataset can be available in dataset folder. The data format is as follows('\t' means TAB):

```
工程 的 立项 、 设计 都 要 充分 论证 , 反复 比较 , 使 工程 经 得 起 时间 检验 。 \t <bos> the establishment and design of projects must be fully debated and repeated comparisons must be made in order to make the project stand the test of time . <eos>
...
```
Note: <'bos> and <'eos> represent the start and end of the target language respectively.

## How to run
#### Train & Dev
Original training dataset is randomly split into 90% for training and 10% for dev.
```
$  python main.py --train_data_path dataset/train.tsv --lr 2e-4 --train_batch_size 128
```
#### Test Interaction
注： transformer_nmt_Model.pt 为在验证集（训练集的 10%用于验证调参）上的最优模型，可从[百度云](https://pan.baidu.com/s/1aAKeYNp-DBCiYVtB6EafLg)下载，并将其放在results文件夹下
```
$  python main.py --train_data_path dataset/train.tsv --load_model results/transformer_nmt_Model.pt
```

More detailed configurations can be found in `config.py`, which is in utils folder.
