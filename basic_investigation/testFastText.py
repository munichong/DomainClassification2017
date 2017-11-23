'''
Created on Oct 27, 2017

@author: munichong

https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27
'''
import time
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

start_time = time.time()

# Creating the model
en_model = FastText.load_fasttext_format('../FastText/wiki.en/wiki.en')

# Getting the tokens 
# words = []
# for word in en_model.vocab:
#     words.append(word)
# 
# # Printing out number of tokens available
# print("Number of Tokens: {}".format(len(words)))
# 
# # Printing out the dimension of a word vector 
# print("Dimension of a word vector: {}".format(
#     len(en_model[words[0]])
# ))

# Print out the vector of a word 
print("Vector components of a word: {}".format(
    en_model["carcaryou"]
))

if '18424' in en_model:
    print("Yes")
else:
    print("No")

if 'carcaryou' in en_model:
    print("Yes")
else:
    print("No")

if '4010' in en_model:
    print("Yes")
else:
    print("No")


runtime = time.time() - start_time
print("--- TOTAL RUNTIME: %d hours %d minutes %d seconds ---" % (runtime // 3600 % 60, runtime // 60 % 60, runtime % 60))
