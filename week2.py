# Importing packages and loading in the data set
from utils_pos import get_word_tag, preprocess
import pandas as pd
from collections import defaultdict
import math
import numpy as np

# load in the training corpus
with open("data/WSJ_02-21.pos", 'r') as f:
    training_corpus = f.readlines()

print(f"A few items of the training corpus list")
print(training_corpus[0:5])

# read the vocabulary data, split by each line of text, and save the list
with open("data/hmm_vocab.txt", 'r') as f:
    voc_l = f.read().split('\n')

print("A few items of the vocabulary list")
print(voc_l[0:50])
print()
print("A few items at the end of the vocabulary list")
print(voc_l[-50:])

# vocab: dictionary that has the index of the corresponding words
vocab = {}

# Get the index of the corresponding words.
for i, word in enumerate(sorted(voc_l)):
    vocab[word] = i

print("Vocabulary dictionary, key is the word, value is a unique integer")
cnt = 0
for k, v in vocab.items():
    print(f"{k}:{v}")
    cnt += 1
    if cnt > 20:
        break

# load in the test corpus
with open("data/WSJ_24.pos", 'r') as f:
    y = f.readlines()

print("A sample of the test corpus")
print(y[0:10])

#corpus without tags, preprocessed
_, prep = preprocess(vocab, "data/test.words")

print('The length of the preprocessed test corpus: ', len(prep))
print('This is a sample of the test_corpus: ')
print(prep[0:10])