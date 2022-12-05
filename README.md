# Replicating Text GCN:

The replication of Text GCN from:

Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377

In this project, we utilize Text GCN to detect whether a news article's headline contains clickbait or not.

## Requirements

Please make sure to launch the Docker image through `launch-scipy-ml.sh -i srgelinas/dsc180a_m5 -s`

## Reproducing Results

Run `python run.py test` to test Text GCN on a small subset of the `clickbait` dataset.

Change `test` in the above command line to your datafile name of choice (e.g. `clickbait`) when producing results for other datasets.

The `clickbait` dataset is available [here](https://drive.google.com/file/d/1Og_EdQSQj_dzYp9ggmdjbdL_X3Q3eIE8/view?usp=sharing).

## Example input data

1. `/test/testdata/test.txt` indicates headline names, training/test split, headline labels. Each line is for a headline.

2. `/data/corpus/test.txt` contains raw text of each headline, each line is for the corresponding line in `/data/test.txt`

3. `prepare_data.py` is an example for preparing your own data, note that '\n' is removed in your documents or sentences.