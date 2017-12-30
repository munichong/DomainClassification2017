'''
Created on Oct 30, 2017

@author: munichong
'''
import numpy as np, pickle, json, csv, os
import tensorflow as tf
from tensorflow.contrib import slim
from pprint import pprint
from random import shuffle
from tabulate import tabulate
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
import warnings
import posttrain_fasttext_classifier_with_desc as pfc

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.wrappers import FastText

DATASET = 'content'  # 'content' or '2340768'

char_ngram = 4

domain_network_type = pfc.domain_network_type
desc_network_type = pfc.desc_network_type
# For RNN
n_rnn_neurons = 300
# For CNN
domain_filter_sizes = [2,1]
desc_filter_sizes = pfc.desc_filter_sizes
domain_num_filters = 512
desc_num_filters = pfc.desc_num_filters

char_embed_dimen = 50
word_embed_dimen = pfc.word_embed_dimen

dropout_rate= 0.0  # dropout leads to worse performance
n_fc_layers_domain = 3
width_fc_layers_domain = 1200
n_fc_layers_desc= pfc.n_fc_layers_desc
width_fc_layers_desc = pfc.width_fc_layers_desc
width_final_rep = pfc.width_fc_layers_desc

act_fn = tf.nn.relu

truncated_desc_words_len = pfc.truncated_desc_words_len


calibration_reg_factor = 0.00  # regularization leads to worse performance

n_epochs = 20
batch_size = 64
autoencoder_lr_rate = 0.001

class_weighted = False


OUTPUT_DIR = pfc.OUTPUT_DIR

category2index = pfc.category2index
categories = pfc.categories

# Creating the model
# print("Loading the FastText Model")
# en_model = {"test":np.array([0]*300)}
# en_model = FastText.load_fasttext_format('../FastText/wiki.en/wiki.en')



class domain_desc_calibrator:

    def __init__(self):
        ''' load data '''
        origin_train_domains = pickle.load(open(OUTPUT_DIR + 'training_domains_%s.list' % DATASET, 'rb'))
        self.domains_train = []
        self.domains_val = []
        self.domains_test = []
        for domains in origin_train_domains:
            shuffle(domains)
            train_end_index = int(len(domains) * 0.8)
            self.domains_train.append(domains[ : train_end_index])
            validation_end_index = int(len(domains) * (0.8 + (1 - 0.8) / 2))
            self.domains_val.append(domains[train_end_index : validation_end_index] )
            self.domains_test.append(domains[validation_end_index: ])

        # self.domains_train = pickle.load(open(OUTPUT_DIR + 'training_domains_%s.list' % DATASET, 'rb'))
        self.domains_train = [d for cat_domains in self.domains_train for d in cat_domains]
        # self.domains_val = pickle.load(open(OUTPUT_DIR + 'validation_domains_%s.list' % DATASET, 'rb'))
        self.domains_val = [d for cat_domains in self.domains_val for d in cat_domains]
        # self.domains_test = pickle.load(open(OUTPUT_DIR + 'test_domains_%s.list' % DATASET, 'rb'))
        self.domains_test = [d for cat_domains in self.domains_test for d in cat_domains]
        print(len(self.domains_train), len(self.domains_val), len(self.domains_test))

        self.charngram2index = pickle.load(open(os.path.join(OUTPUT_DIR, 'charngram2index.dict'), 'rb'))
        self.word2index = pickle.load(open(os.path.join(OUTPUT_DIR, 'word2index.dict'), 'rb'))

        ''' load params '''
        self.params = json.load(open(OUTPUT_DIR + 'params_%s.json' % DATASET))
        self.params['max_desc_words_len'] = min(truncated_desc_words_len, self.params['max_desc_words_len'])
        self.compute_class_weights()

    def compute_class_weights(self):
        n_total = sum(self.params['category_dist_traintest'].values())
        n_class = len(self.params['category_dist_traintest'])
        self.class_weights = {cat: max(min( n_total / (n_class * self.params['category_dist_traintest'][cat]), 1.5), 0.5)
                              for cat, size in self.params['category_dist_traintest'].items()}
        # self.class_weights['Sports'] = 1
        # self.class_weights['Health'] = 1
        # self.class_weights['Business'] = 0.8
        # self.class_weights['Arts'] = 0.8
        pprint(self.class_weights)

    def next_batch(self, domains, batch_size=batch_size):
        X_batch_domain = []
        X_batch_domain_mask = []
        domain_actual_lens = []
        X_batch_desc = []
        X_batch_desc_mask = []
        desc_actual_lens = []
        sample_weights = []
        y_batch = []
        shuffle(domains)
        start_index = 0
        while start_index < len(domains):
            for i in range(start_index, min(len(domains), start_index + batch_size)):
                ''' get char n-gram indices '''
                embeds = []  # [[1,2,5,0,0], [35,3,7,8,4], ...]
                # print(domains[i])
                for word in domains[i]['segmented_domain']:
                    embeds.append([self.charngram2index[word[start : start + char_ngram]] for start in range(max(1, len(word) - char_ngram))])
                domain_actual_lens.append(len(embeds))
                ''' domain char n-gram padding '''
                # pad char-ngram level
                embeds = [indices + [0] * (self.params['max_segment_char_len'] - char_ngram + 1 - len(indices))for indices in embeds]
                # pad segment level
                embeds += [[0] * (self.params['max_segment_char_len'] - char_ngram + 1) for _ in range(self.params['max_domain_segments_len'] - len(embeds))]
                X_batch_domain.append(embeds)
                ''' domain char n-gram mask '''
                X_batch_domain_mask.append((np.array(embeds) != 0).astype(float))


                ''' description '''
                desc_indice = [self.word2index[word.lower()] for word in domains[i]['tokenized_desc'][ : self.params['max_desc_words_len']]]  # truncate
                desc_actual_lens.append(len(desc_indice))
                ''' description padding '''
                if self.params['max_desc_words_len'] >= len(desc_indice):
                    desc_indice += [0] * (self.params['max_desc_words_len'] - len(desc_indice))
                else:
                    desc_indice += [0] * (self.params['max_desc_words_len'] - len(desc_indice))
                X_batch_desc.append(desc_indice)
                # assert len(desc_indice) == self.params['max_desc_words_len']
                ''' description mask '''
                X_batch_desc_mask.append((np.array(desc_indice) != 0).astype(float))


                ''' target category '''
                sample_weights.append(self.class_weights[categories[domains[i]['target']]])
                y_batch.append(domains[i]['target'])

            # print(np.array(X_batch_desc).shape)
            # print(np.array(X_batch_desc_mask).shape)
            # print(X_batch_desc[:10])
            # print(X_batch_desc_mask[:10])

            yield np.array(X_batch_domain), np.array(X_batch_domain_mask), np.array(domain_actual_lens), \
                  np.array(X_batch_desc), np.array(X_batch_desc_mask), np.array(desc_actual_lens), \
                  np.array(sample_weights), np.array(y_batch)

            # print(sample_weights)


            X_batch_domain.clear()
            X_batch_domain_mask.clear()
            domain_actual_lens.clear()
            X_batch_desc.clear()
            X_batch_desc_mask.clear()
            desc_actual_lens.clear()
            sample_weights.clear()
            y_batch.clear()
            start_index += batch_size


    def evaluate(self, data, session, eval_nodes):
        total_loss = 0
        n_batch = 0
        for X_batch_domain, X_batch_domain_mask, domain_actual_lens, \
            X_batch_desc, X_batch_desc_mask, desc_actual_lens, \
            sample_weights, y_batch in self.next_batch(data):

            batch_loss, = session.run(eval_nodes,
                                              feed_dict={
                                                        'bool_train:0': True,
                                                        'domain_input:0': X_batch_domain,
                                                        'domain_mask:0': X_batch_domain_mask,
                                                        'domain_length:0': domain_actual_lens,
                                                        'desc_input:0': X_batch_desc,
                                                        'desc_mask:0': X_batch_desc_mask,
                                                        'desc_length:0': desc_actual_lens,
                                                        'weight:0': sample_weights,
                                                        'target:0': y_batch})
            total_loss += batch_loss
            n_batch += 1
        return total_loss / n_batch



    def get_rnn_output(self, embed, seq_len, is_training, trainable=True):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(n_rnn_neurons, activation=tf.nn.tanh)
        # The shape of last_states should be [batch_size, n_lstm_neurons]
        _, rnn_output = tf.nn.dynamic_rnn(rnn_cell, embed, sequence_length=seq_len, dtype=tf.float32, time_major=False)
        rnn_output = tf.layers.dropout(rnn_output, dropout_rate, training=is_training)
        return rnn_output

    def get_cnn_output(self, embed_dimen, embed, max_seq_len, num_filters, filter_sizes, is_training, trainable=True):
        pooled_outputs = []
        for filter_size in filter_sizes:
            # Define and initialize filters
            filter_shape = [filter_size, embed_dimen, 1, num_filters]
            W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), trainable=trainable) # initialize the filters' weights
            b_filter = tf.Variable(tf.constant(0.1, shape=[num_filters]), trainable=trainable)  # initialize the filters' biases
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


    def run_graph(self):
        ''' INPUT '''
        is_training = tf.placeholder(tf.bool, shape=(), name='bool_train')
        # the input can be word indices or pre-trained FastText vectors
        # the input should be padded.
        x_desc = tf.placeholder(tf.int32, shape=[None, self.params['max_desc_words_len']], name='desc_input')
        x_domain = tf.placeholder(tf.int32,shape=[None, self.params['max_domain_segments_len'],
                                        self.params['max_segment_char_len'] - char_ngram + 1], name='domain_input')
        desc_len = tf.placeholder(tf.int32, shape=[None], name='desc_length')
        domain_len = tf.placeholder(tf.int32, shape=[None], name='domain_length')

        sample_weights = tf.placeholder(tf.float32, shape=[None], name='weight')
        y = tf.placeholder(tf.int32, shape=[None],
                           name='target')  # Each entry in y must be an index in [0, num_classes)


        ''' Abstractize Descriptions (Should be all non-trainable) '''
        # embedding layers
        desc_embeddings = tf.Variable(tf.random_uniform([len(self.word2index), word_embed_dimen], -1.0, 1.0),
                                      trainable=False)
        desc_embed = tf.nn.embedding_lookup(desc_embeddings, x_desc)
        desc_mask = tf.placeholder(tf.float32, shape=[None, self.params['max_desc_words_len']], name='desc_mask')
        desc_mask = tf.expand_dims(desc_mask, axis=-1)
        desc_mask = tf.tile(desc_mask, [1,1,word_embed_dimen])
        x_embed_desc = tf.multiply(desc_embed, desc_mask)

        desc_vectors = []
        # if 'RNN' in type:
        #     domain_vec_rnn = self.get_rnn_output(desc_word_embed, desc_len, is_training)
        #     desc_vectors.append(domain_vec_rnn)
        if 'CNN' in desc_network_type:
            with tf.variable_scope('cnn_desc'):
                desc_vec_cnn = self.get_cnn_output(word_embed_dimen, x_embed_desc,
                                            self.params['max_desc_words_len'], desc_num_filters, desc_filter_sizes, is_training, trainable=False)

                # for _ in range(n_fc_layers_desc):
                for _ in range(n_fc_layers_desc - 1):
                    logits_desc = tf.contrib.layers.fully_connected(desc_vec_cnn, num_outputs=width_fc_layers_desc,
                                                                activation_fn=act_fn, trainable=False)
                    logits_desc = tf.layers.dropout(logits_desc, dropout_rate, training=is_training)

                # logits_desc = tf.contrib.layers.fully_connected(logits_desc, num_outputs=width_final_rep,
                #                                                 activation_fn=act_fn, trainable=False)
                # logits_desc = tf.layers.dropout(logits_desc, dropout_rate, training=is_training)

            desc_vectors.append(logits_desc)


        cat_layer_desc = tf.concat(desc_vectors, -1)





        ''' Abstractize Domains '''
        # embedding layers
        domain_embeddings = tf.Variable(tf.random_uniform([len(self.charngram2index), char_embed_dimen], -1.0, 1.0))
        domain_embed = tf.nn.embedding_lookup(domain_embeddings, x_domain)
        domain_mask = tf.placeholder(tf.float32, shape=[None, self.params['max_domain_segments_len'],
                                                 self.params['max_segment_char_len'] - char_ngram + 1], name='domain_mask')
        domain_mask = tf.expand_dims(domain_mask, axis=-1)
        domain_mask = tf.tile(domain_mask, [1,1,1,char_embed_dimen])
        domain_embed = tf.multiply(domain_embed, domain_mask)
        domain_embed = tf.reduce_mean(domain_embed, 2)

        domain_vectors = []
        # if 'RNN' in domain_network_type:
        #     domain_vec_rnn = self.get_rnn_output(domain_char_embed, domain_len, is_training)
        #     domain_vectors.append(domain_vec_rnn)
        if 'CNN' in domain_network_type:
            with tf.variable_scope('cnn_domain'):
                domain_vec_cnn = self.get_cnn_output(char_embed_dimen, domain_embed,
                                            self.params['max_domain_segments_len'], domain_num_filters, domain_filter_sizes, is_training)

                for _ in range(n_fc_layers_domain - 1):
                    logits_domain = tf.contrib.layers.fully_connected(domain_vec_cnn, num_outputs=width_fc_layers_domain,
                                                                activation_fn=act_fn)
                    logits_domain = tf.layers.dropout(logits_domain, dropout_rate, training=is_training)

                logits_domain = tf.contrib.layers.fully_connected(logits_domain, num_outputs=width_final_rep,
                                                                  activation_fn=act_fn)
                logits_domain = tf.layers.dropout(logits_domain, dropout_rate, training=is_training)

            domain_vectors.append(logits_domain)

        cat_layer_domain = tf.concat(domain_vectors, -1)



        reconstruction_loss = tf.losses.mean_squared_error(cat_layer_desc, cat_layer_domain, weights=1.0)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = reconstruction_loss + calibration_reg_factor * sum(reg_losses)

        optimizer = tf.train.AdamOptimizer(autoencoder_lr_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        print('Trainable Variables:')
        print(tf.trainable_variables())

        variables_to_restore = {v.name: v for v in tf.global_variables() if v.name.split('/')[0] == 'cnn_desc'}
        print("variables_to_restore:", ['desc_embeddings'] + sorted(variables_to_restore.keys()))
        saver_for_desc_restore = tf.train.Saver({**{"desc_embeddings" : desc_embeddings}, **variables_to_restore})

        variables_to_save = {v.name: v for v in tf.global_variables() if v.name.split('/')[0] == 'cnn_domain'}
        print("variables_to_save:", ['domain_embeddings'] + sorted(variables_to_save.keys()))
        saver_for_domain_save = tf.train.Saver({**{'domain_embeddings' : domain_embeddings}, **variables_to_save})

        ''' Make sure all variables about desc CNN are non-trainable '''
        assert [] == [v for v in tf.trainable_variables() if v.name.split('/')[0] == 'cnn_desc']

        with tf.Session() as sess:
            init.run()

            # Restore variables from disk.
            saver_for_desc_restore.restore(sess, os.path.join(OUTPUT_DIR, 'desc_abstraction.params'))
            print("Model restored.")
            # print(desc_embeddings.eval())

            n_total_batches = int(np.ceil(len(self.domains_train) / batch_size))
            test_loss_history = []
            for epoch in range(1, n_epochs + 1):
                # model training
                n_batch = 0
                for X_batch_domain, X_batch_domain_mask, domain_actual_lens, \
                    X_batch_desc, X_batch_desc_mask, desc_actual_lens, \
                    sample_weights, y_batch in self.next_batch(self.domains_train):
                    _, loss_batch_train = sess.run([training_op, loss],
                                                    feed_dict={
                                                        'bool_train:0': True,
                                                        'domain_input:0': X_batch_domain,
                                                        'domain_mask:0': X_batch_domain_mask,
                                                        'domain_length:0': domain_actual_lens,
                                                        'desc_input:0': X_batch_desc,
                                                        'desc_mask:0': X_batch_desc_mask,
                                                        'desc_length:0': desc_actual_lens,
                                                        'weight:0': sample_weights,
                                                        'target:0': y_batch})

                    n_batch += 1
                    if epoch < 2:
                        # print(prediction_train)
                        print("Epoch %d - Batch %d/%d: loss = %.4f" %
                              (epoch, n_batch, n_total_batches, loss_batch_train))



                ''''''''''''''''''''''''''''''''''''
                ''' evaluation on training data '''
                ''''''''''''''''''''''''''''''''''''
                eval_nodes = [loss]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_train = self.evaluate(self.domains_train, sess, eval_nodes)
                print("*** On Training Set:\tloss = %.6f" % (loss_train))

                ''''''''''''''''''''''''''''''''''''''
                ''' evaluation on validation data '''
                ''''''''''''''''''''''''''''''''''''''
                loss_val = self.evaluate(self.domains_val, sess, eval_nodes)
                print("*** On Validation Set:\tloss = %.6f" % (loss_val))

                ''''''''''''''''''''''''''''''''''''''
                ''' evaluation on test data '''
                ''''''''''''''''''''''''''''''''''''''
                loss_test = self.evaluate(self.domains_test, sess, eval_nodes)
                print("*** On Test Set:\tloss = %.6f" % (loss_test))


                if not test_loss_history or loss_test < min(test_loss_history):
                    saver_for_domain_save.save(sess, os.path.join(OUTPUT_DIR, 'domain_abstraction.params'))
                    print('Save on the disk at %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                test_loss_history.append(loss_test)



if __name__ == '__main__':
    calibrator = domain_desc_calibrator()
    calibrator.run_graph()
