import numpy as np, pickle, json, csv, os
import tensorflow as tf
from pprint import pprint
from random import shuffle
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.wrappers import FastText
from gensim.models.wrappers.fasttext import compute_ngrams, FastTextKeyedVectors


'''
print(sorted(compute_ngrams("www", 3, 6)))
print(sorted(compute_ngrams("npr", 3, 6)))
a = []
word = 'aaaaaa'
word = ''.join(['<', word, '>'])
for size in [4, 5]:
    for i in range(max(1, len(word) - size + 1)):  # some segments' lengths are less than char_ngram
        a.append(word[i : i + size])
print(sorted(a))
'''





print("Loading the FastText Model")
# en_model = {"test":np.array([0]*300)}
en_model = FastText.load_fasttext_format('../FastText/wiki.en/wiki.en')
print(type(en_model))



def ft_embed(word):
    if word in en_model.wv.vocab:
        return super(FastTextKeyedVectors, en_model.wv).word_vec(word)

    word_vec = np.zeros(en_model.wv.syn0_ngrams.shape[1], dtype=np.float32)
    ngrams = compute_ngrams(word, 3, 6)
    ngrams = [ng for ng in ngrams if ng in en_model.wv.ngrams]
    ngram_weights = en_model.wv.syn0_ngrams
    for ngram in ngrams:
        word_vec += ngram_weights[en_model.wv.ngrams[ngram]]
    if word_vec.any():
        return word_vec / len(ngrams)

print(en_model['wwwreal'] == ft_embed('wwwreal'))
print(en_model['real'] == ft_embed('real'))