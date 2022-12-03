#!/usr/bin/python
#-*-coding:utf-8-*-

dataset_name = 'new_dataset'
sentences = ['This is sentence 1.​', 'This is sentence 2.']
labels = ['Yes' , 'No']
train_or_test_list = ['train', 'test']

meta_data_list = []

for i in range(len(sentences)):
    meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
    meta_data_list.append(meta)

meta_data_str = '\n'.join(meta_data_list)

f = open('data/' + dataset_name + '.txt', 'w')
f.write(meta_data_str)
f.close()

corpus_str = '\n'.join(sentences)

f = open('data/corpus/' + dataset_name + '.txt', 'w')
f.write(corpus_str)
f.close()