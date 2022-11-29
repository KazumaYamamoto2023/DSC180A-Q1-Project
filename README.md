# Replicating text_gcn

The replication of Text GCN from:

Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377

## Requirements

Python 2.7 or 3.6

Tensorflow >= 1.4.0

## Reproducing Results

Run `python run.py clickbait`

Change `clickbait` in the above command line to `20ng`, `R8`, `R52`, `ohsumed` and `mr` when producing results for other datasets.

## Example input data

1. `/data/clickbait.txt` indicates headline names, training/test split, headline labels. Each line is for a headline.

2. `/data/corpus/clickbait.txt` contains raw text of each headline, each line is for the corresponding line in `/data/clickbait.txt`

3. `prepare_data.py` is an example for preparing your own data, note that '\n' is removed in your documents or sentences.