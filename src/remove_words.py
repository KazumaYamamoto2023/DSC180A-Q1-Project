from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import sys
import os
cwd = os.getcwd()
sys.path.insert(0, cwd + '/src/')
from utils import clean_str, loadWord2Vec
sys.path.insert(0, cwd)


def clean(dataset):
    num = 5
    if dataset == 'test':
        num = 1
    
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    print(stop_words)

    # Read Word Vectors
    # word_vector_file = 'data/glove.6B/glove.6B.200d.txt'
    # vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)
    # word_embeddings_dim = len(embd[0])
    # dataset = '20ng'

    doc_content_list = []
    if dataset == 'test':
        f = open('test/testdata/corpus/' + dataset + '.txt', 'rb')
    else:
        f = open('data/corpus/' + dataset + '.txt', 'rb')
    # f = open('data/wiki_long_abstracts_en_text.txt', 'r')
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))
    f.close()


    word_freq = {}  # to remove rare words

    for doc_content in doc_content_list:
        temp = clean_str(doc_content)
        words = temp.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    clean_docs = []
    for doc_content in doc_content_list:
        temp = clean_str(doc_content)
        words = temp.split()
        doc_words = []
        for word in words:
            # word not in stop_words and word_freq[word] >= 5
            if dataset == 'mr':
                doc_words.append(word)
            elif word not in stop_words and word_freq[word] >= 5:
                doc_words.append(word)

        doc_str = ' '.join(doc_words).strip()
        #if doc_str == '':
            #doc_str = temp
        clean_docs.append(doc_str)

    clean_corpus_str = '\n'.join(clean_docs)

    if dataset == 'test':
        f = open('test/testdata/corpus/' + dataset + '.clean.txt', 'rb')
    else:
        f = open('data/corpus/' + dataset + '.clean.txt', 'w')
    #f = open('data/wiki_long_abstracts_en_text.clean.txt', 'w')
    f.write(clean_corpus_str)
    f.close()

    #dataset = '20ng'
    min_len = 10000
    aver_len = 0
    max_len = 0 

    if dataset == 'test':
        f = open('test/testdata/corpus/' + dataset + '.clean.txt', 'rb')
    else:
        f = open('data/corpus/' + dataset + '.clean.txt', 'r')
    #f = open('data/wiki_long_abstracts_en_text.txt', 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        temp = line.split()
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    f.close()
    aver_len = 1.0 * aver_len / len(lines)
    print('min_len : ' + str(min_len))
    print('max_len : ' + str(max_len))
    print('average_len : ' + str(aver_len))