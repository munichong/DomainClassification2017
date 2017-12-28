'''
Created on Oct 30, 2017

@author: munichong
'''
import numpy as np, pickle, json, csv, os
import tensorflow as tf
from pprint import pprint
from random import shuffle
from tabulate import tabulate
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.wrappers import FastText

DATASET = 'content'  # 'content' or '2340768'

char_ngram = 4

autoencoder_type = 'CNN'
# For RNN
n_rnn_neurons = 300
# For CNN
filter_sizes = [2,1]
num_filters = 512

desc_word_embed_dimen = 100
domain_char_embed_dimen = 50
# n_fc_neurons = 64
dropout_rate= 0.2
n_fc_layers_desc= 3
n_fc_layers_doamin = 3
act_fn = tf.nn.relu

autoencoder_reg_factor = 0.001

n_epochs = 100
batch_size = 4000
autoencoder_lr_rate = 0.001

class_weighted = False


OUTPUT_DIR = '../Output/'

category2index = pickle.load(open(os.path.join(OUTPUT_DIR + 'category2index_%s.dict' % DATASET), 'rb'))
categories = [''] * len(category2index)
for cate, i in category2index.items():
    categories[i] = cate
print(categories)

# Creating the model
print("Loading the FastText Model")
# en_model = {"test":np.array([0]*300)}
# en_model = FastText.load_fasttext_format('../FastText/wiki.en/wiki.en')



class PosttrainFastTextDescClassifier:

    def __init__(self):
        ''' load data '''
        self.domains_train = pickle.load(open(OUTPUT_DIR + 'training_domains_%s.list' % DATASET, 'rb'))
        self.domains_train = [d for cat_domains in self.domains_train for d in cat_domains]
        self.domains_val = pickle.load(open(OUTPUT_DIR + 'validation_domains_%s.list' % DATASET, 'rb'))
        self.domains_val = [d for cat_domains in self.domains_val for d in cat_domains]
        self.domains_test = pickle.load(open(OUTPUT_DIR + 'test_domains_%s.list' % DATASET, 'rb'))
        self.domains_test = [d for cat_domains in self.domains_test for d in cat_domains]

        self.charngram2index = defaultdict(lambda: len(self.charngram2index) + 1)  # index starts from 1. 0 is for padding
        for domains in (self.domains_train, self.domains_val, self.domains_test):
            for domain in domains:
                for word in domain['segmented_domain']:
                    for i in range(max(1, len(word) - char_ngram)):  # some segments' lengths are less than char_ngram
                        self.charngram2index[word[i : i + char_ngram]]

        ''' load params '''
        self.params = json.load(open(OUTPUT_DIR + 'params_%s.json' % DATASET))


    def get_rnn_output(self, embed, seq_len, is_training):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(n_rnn_neurons, activation=tf.nn.tanh)
        # The shape of last_states should be [batch_size, n_lstm_neurons]
        _, rnn_output = tf.nn.dynamic_rnn(rnn_cell, embed, sequence_length=seq_len, dtype=tf.float32, time_major=False)
        rnn_output = tf.layers.dropout(rnn_output, dropout_rate, training=is_training)
        return rnn_output

    def get_cnn_output(self, embed_dimen, embed, max_seq_len, is_training):
        pooled_outputs = []
        for filter_size in filter_sizes:
            # Define and initialize filters
            filter_shape = [filter_size, embed_dimen, 1, num_filters]
            W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1)) # initialize the filters' weights
            b_filter = tf.Variable(tf.constant(0.1, shape=[num_filters]))  # initialize the filters' biases
            # The conv2d operation expects a 4-D tensor with dimensions corresponding to batch, width, height and channel.
            # The result of our embedding doesnâ€™t contain the channel dimension
            # So we add it manually, leaving us with a layer of shape [None, sequence_length, embedding_size, 1].
            x_embed_expanded = tf.expand_dims(embed, -1)
            conv = tf.nn.conv2d(x_embed_expanded, W_filter, strides=[1, 1, 1, 1], padding="VALID")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b_filter), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, max_seq_len - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1], padding='VALID')
            pooled_outputs.append(pooled)
        # Combine all the pooled features
        h_pool = tf.concat(pooled_outputs, axis=3)
        num_filters_total = num_filters * len(filter_sizes)
        cnn_output = tf.reshape(h_pool, [-1, num_filters_total])
        cnn_output = tf.layers.dropout(cnn_output, dropout_rate, training=is_training)
        return cnn_output


    def run_autoencoder(self):
        ''' INPUT '''
        is_training = tf.placeholder(tf.bool, shape=(), name='bool_train')
        # the input can be word indices or pre-trained FastText vectors
        # the input should be padded.
        x_desc = tf.placeholder(tf.int32, shape=[None, self.params['max_desc_words_len']], name='desc_input')
        x_domain = tf.placeholder(tf.int32,shape=[None, self.params['max_domain_segments_len'],
                                        self.params['max_segment_char_len'] - char_ngram + 1], name='domain_input')
        desc_len = tf.placeholder(tf.int32, shape=[None], name='desc_length')
        domain_len = tf.placeholder(tf.int32, shape=[None], name='domain_length')


        ''' Abstractize Descriptions '''
        # embedding layers
        desc_word_embeddings = tf.Variable(tf.random_uniform([len(self.descwords2index), desc_word_embed_dimen], -1.0, 1.0))
        desc_word_embed = tf.nn.embedding_lookup(desc_word_embeddings, x_desc)
        desc_word_mask = tf.placeholder(tf.float32, shape=[None, self.params['max_desc_words_len']], name='desc_mask')
        desc_word_mask = tf.expand_dims(desc_word_mask, axis=-1)
        desc_word_mask = tf.tile(desc_word_mask, [1,1,desc_word_embed_dimen])
        desc_word_embed = tf.multiply(desc_word_embed, desc_word_mask)

        desc_vectors = []
        if 'RNN' in type:
            domain_vec_rnn = get_rnn_output(desc_word_embed, desc_len, is_training)
            desc_vectors.append(domain_vec_rnn)
        if 'CNN' in type:
            domain_vec_cnn = get_cnn_output(desc_word_embed_dimen, desc_word_embed,
                                            self.params['max_desc_words_len'], is_training)
            desc_vectors.append(domain_vec_cnn)

        cat_layer_desc = tf.concat(desc_vectors, -1)

        for _ in range(n_fc_layers_desc):
            logits_desc = tf.contrib.layers.fully_connected(cat_layer_desc, num_outputs=n_rnn_neurons, activation_fn=act_fn)
            logits_desc = tf.layers.dropout(logits_desc, dropout_rate, training=is_training)



        ''' Abstractize Domains '''
        # embedding layers
        domain_char_embeddings = tf.Variable(tf.random_uniform([len(self.charngram2index), domain_char_embed_dimen], -1.0, 1.0))
        domain_char_embed = tf.nn.embedding_lookup(domain_char_embeddings, x_domain)
        domain_char_mask = tf.placeholder(tf.float32, shape=[None, self.params['max_domain_segments_len'],
                                                 self.params['max_segment_char_len'] - char_ngram + 1], name='domain_mask')
        domain_char_mask = tf.expand_dims(domain_char_mask, axis=-1)
        domain_char_mask = tf.tile(domain_char_mask, [1,1,1,domain_char_embed_dimen])
        domain_char_embed = tf.multiply(domain_char_embed, domain_char_mask)
        domain_char_embed = tf.reduce_sum(domain_char_embed, 2)

        domain_vectors = []
        if 'RNN' in type:
            domain_vec_rnn = get_rnn_output(domain_char_embed, domain_len, is_training)
            desc_vectors.append(domain_vec_rnn)
        if 'CNN' in type:
            domain_vec_cnn = get_cnn_output(domain_char_embed_dimen, domain_char_embed,
                                            self.params['max_domain_segments_len'], is_training)
            desc_vectors.append(domain_vec_cnn)

        cat_layer_domain = tf.concat(domain_vectors, -1)

        for _ in range(n_fc_layers_domain):
            logits_domain = tf.contrib.layers.fully_connected(cat_layer_domain, num_outputs=n_rnn_neurons, activation_fn=act_fn)
            logits_domain = tf.layers.dropout(logits_domain, dropout_rate, training=is_training)

        reconstruction_loss = tf.reduce_mean(tf.square(logits_desc, logits_domain))  # MSE

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = my_normal_loss + autoencoder_reg_factor * sum(reg_losses)

        optimizer = tf.train.AdamOptimier(autoencoder_lr_rate)
        train_op = optimizer.minimize(loss)

        init = tf.global_veriable_initializer()



        with tf.Session() as sess:
            init.run()
            n_total_batches = int(np.ceil(len(self.domains_train) / batch_size))








if __name__ == '__main__':
    classifier = PosttrainFastTextDescClassifier()
    classifier.run_autoencoder()
