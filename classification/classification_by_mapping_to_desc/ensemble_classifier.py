'''
Created on Oct 30, 2017

@author: munichong
'''
import numpy as np, pickle, json, csv, os
import tensorflow as tf
from collections import defaultdict
from random import shuffle
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.wrappers import FastText
from gensim.models.wrappers.fasttext import compute_ngrams

DATASET = 'content'  # 'content' or '2340768'

char_ngram_sizes = [4,5]

type = 'CNN'
# For RNN
n_rnn_neurons = 300
# For CNN
filter_sizes = [2,1]
num_filters = 512

embed_dimen = 300
# n_fc_neurons = 64
dropout_rate= 0.2
n_fc_layers= 3
act_fn = tf.nn.relu

n_epochs = 80
batch_size = 2000
lr_rate = 0.0001


OUTPUT_DIR = '../../Output/'

category2index = pickle.load(open(os.path.join(OUTPUT_DIR + 'category2index_%s.dict' % DATASET), 'rb'))
categories = [''] * len(category2index)
for cate, i in category2index.items():
    categories[i] = cate
print(categories)

# Creating the model
print("Loading the FastText Model")
# en_model = {"test":np.array([0]*300)}
en_model = FastText.load_fasttext_format('../../FastText/wiki.en/wiki.en')


class PretrainFastTextClassifier:

    def __init__(self):
        ''' load data '''
        self.domains_train = pickle.load(open(OUTPUT_DIR + 'training_domains_%s.list' % DATASET, 'rb'))
        self.domains_train = [d for cat_domains in self.domains_train for d in cat_domains ]
        self.domains_val = pickle.load(open(OUTPUT_DIR + 'validation_domains_%s.list' % DATASET, 'rb'))
        self.domains_val = [d for cat_domains in self.domains_val for d in cat_domains]
        self.domains_test = pickle.load(open(OUTPUT_DIR + 'test_domains_%s.list' % DATASET, 'rb'))
        self.domains_test = [d for cat_domains in self.domains_test for d in cat_domains]

        self.charngram2index = defaultdict(int)  # index starts from 1. 0 is for padding
        for domains in (self.domains_train, self.domains_val, self.domains_test):
            for domain in domains:
                for word in domain['segmented_domain']:

                    for ngram in compute_ngrams(word, *char_ngram_sizes):
                        if ngram in self.charngram2index:
                            continue
                        self.charngram2index[ngram] = len(self.charngram2index) + 1

        ''' load params '''
        self.params = json.load(open(OUTPUT_DIR + 'params_%s.json' % DATASET))
        self.params['max_segment_char_len'] += 2  # because '<' and '>'are appended to each word
        # the word itself is also added, thus: sum(...) + 1
        self.max_num_charngrams = len(
            compute_ngrams(''.join(['a'] * self.params['max_segment_char_len']), *char_ngram_sizes))

    def next_batch(self, domains, batch_size=batch_size):
        X_batch_embed = []
        X_batch_weighting = []
        X_batch_suf = []
        domain_actual_lens = []
        y_batch = []
        shuffle(domains)
        start_index = 0
        while start_index < len(domains):
            for i in range(start_index, min(len(domains), start_index + batch_size)):
                # skip if a segment is not in en_model
                embeds = [en_model[w].tolist() for w in domains[i]['segmented_domain'] if w in en_model]
                # if not embeds: # Skip if none of segments of this domain can not be recognized by FastText
                #     continue
                domain_actual_lens.append(len(embeds))
                n_extra_padding = self.params['max_domain_segments_len'] - len(embeds)
                embeds += [[0] * embed_dimen for _ in range(n_extra_padding)]
                # X_batch_embed.append(tf.pad(embeds, paddings=[[0, n_extra_padding],[0,0]], mode="CONSTANT"))
                X_batch_embed.append(embeds)


                embeds_weighting = []  # [[1,2,5,0,0], [35,3,7,8,4], ...]
                for word in domains[i]['segmented_domain']:
                    embeds_weighting.append([self.charngram2index[ngram] for ngram in compute_ngrams(word, *char_ngram_sizes)])
                ''' padding '''
                # pad char-ngram level
                embeds_weighting = [indices + [0] * (self.max_num_charngrams - len(indices)) for indices in embeds_weighting]
                embeds_weighting += [[0] * self.max_num_charngrams for _ in
                           range(self.params['max_domain_segments_len'] - len(embeds_weighting))]
                X_batch_weighting.append(embeds_weighting)


                one_hot_suf = np.zeros(self.params['num_suffix'])
                one_hot_suf[domains[i]['suffix_indices']] = 1.0 / len(domains[i]['suffix_indices'])
                X_batch_suf.append(one_hot_suf)

                y_batch.append(domains[i]['target'])
            yield np.array(X_batch_embed), np.array(X_batch_weighting), np.array(domain_actual_lens), np.array(X_batch_suf), \
                  np.array(y_batch)

            # print(sample_weights)

            X_batch_embed.clear()
            X_batch_weighting.clear()
            domain_actual_lens.clear()
            X_batch_suf.clear()
            y_batch.clear()
            start_index += batch_size


    def evaluate(self, data, session, eval_nodes):
        total_correct = 0
        total_loss = 0
        total_bool = []
        total_pred = []
        n_batch = 0
        desc_imp = None
        domain_imp = None
        for X_batch_embed, X_batch_weighting, domain_actual_lens, X_batch_suf, y_batch in self.next_batch(data):
            batch_correct, batch_loss, batch_bool, batch_pred, \
            batch_desc_imp, batch_domain_imp, batch_log1, batch_log2  = session.run(eval_nodes,
                                                         feed_dict={
                                                                    'bool_train:0': False,
                                                                    'embedding:0': X_batch_embed,
                                                                    'indices:0': X_batch_weighting,
                                                                    'suffix:0': X_batch_suf,
                                                                    'length:0': domain_actual_lens,
                                                                    'target:0': y_batch})

            if desc_imp is None:
                print("desc_imp:")
                print(batch_desc_imp)
                print("domain_imp:")
                print(batch_domain_imp)

                print("logits1:")
                print(batch_log1)
                print("logits2:")
                print(batch_log2)
                # print("logits_normal1:")
                # print(batch_lognor1)
                # print("logits_normal2:")
                # print(batch_lognor2)
                # print("logits_softmax1:")
                # print(batch_logsoft1)
                # print("logits_softmax2:")
                # print(batch_logsoft2)


            print("%.1f%%: desc_imp > domain_imp" % (sum( a > b for a, b in zip(batch_desc_imp, batch_domain_imp)) / len(batch_domain_imp) * 100))

            desc_imp = batch_domain_imp
            # domain_imp =batch_domain_imp

            total_loss += batch_loss
            total_correct += batch_correct
            total_bool.extend(batch_bool)
            total_pred.extend(batch_pred)
            n_batch += 1
        return total_loss / n_batch, total_correct / len(data), total_bool, total_pred



    def run_graph(self):

        # tf.reset_default_graph()

        # INPUTs
        is_training = tf.placeholder(tf.bool, shape=(), name='bool_train')
        x_embed = tf.placeholder(tf.float32,
                                 shape=[None, self.params['max_domain_segments_len'], embed_dimen],
                                 name='embedding')

        x_indices = tf.placeholder(tf.int32,
                                 shape=[None, self.params['max_domain_segments_len'],
                                        self.max_num_charngrams],
                                 name='indices')

        embed_dimen_weighting = 300
        num_filter_weighting = 1024
        num_fclayer_width_weighting = 512

        embeddings = tf.Variable(tf.random_uniform([len(self.charngram2index), embed_dimen_weighting], -1.0, 1.0))
        x_embed_weighting = tf.nn.embedding_lookup(embeddings, x_indices)
        x_embed_weighting = tf.reduce_mean(x_embed_weighting, 2)

        # print(x_embed.get_shape())
        x_suffix = tf.placeholder(tf.float32,
                                  shape=[None, self.params['num_suffix']],
                                  name='suffix')

        seq_len = tf.placeholder(tf.int32, shape=[None], name='length')

        y = tf.placeholder(tf.int32, shape=[None], name='target') # Each entry in y must be an index in [0, num_classes)






        with tf.variable_scope('domain_mapping'):

            pooled_outputs = []
            for filter_size in filter_sizes:
                # Define and initialize filters
                filter_shape = [filter_size, embed_dimen, 1, num_filters]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), trainable=False) # initialize the filters' weights
                b_filter = tf.Variable(tf.constant(0.1, shape=[num_filters]), trainable=False)  # initialize the filters' biases
                # The conv2d operation expects a 4-D tensor with dimensions corresponding to batch, width, height and channel.
                # The result of our embedding doesn’t contain the channel dimension
                # So we add it manually, leaving us with a layer of shape [None, sequence_length, embedding_size, 1].
                x_embed_expanded = tf.expand_dims(x_embed, -1)
                conv = tf.nn.conv2d(x_embed_expanded, W_filter, strides=[1, 1, 1, 1], padding="VALID")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b_filter), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.params['max_domain_segments_len'] - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID')
                pooled_outputs.append(pooled)
            # Combine all the pooled features
            h_pool = tf.concat(pooled_outputs, axis=3)
            num_filters_total = num_filters * len(filter_sizes)
            domain_vec_cnn1 = tf.reshape(h_pool, [-1, num_filters_total])
            domain_vec_cnn1 = tf.layers.dropout(domain_vec_cnn1, dropout_rate, training=False)


            logits = domain_vec_cnn1

            for _ in range(n_fc_layers):
                logits = tf.contrib.layers.fully_connected(logits, num_outputs=n_rnn_neurons, activation_fn=act_fn, trainable=False)
                logits = tf.layers.dropout(logits, dropout_rate, training=False)

            logits1 = tf.contrib.layers.fully_connected(logits, self.params['num_targets'], activation_fn=act_fn, trainable=False)

        saver_desc = tf.train.Saver([v for v in tf.all_variables() if 'domain_mapping' in v.name])



        with tf.variable_scope('domain_cnn'):
            domain_vectors = []

            pooled_outputs = []
            for filter_size in filter_sizes:
                # Define and initialize filters
                filter_shape = [filter_size, embed_dimen, 1, num_filters]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), trainable=False) # initialize the filters' weights
                b_filter = tf.Variable(tf.constant(0.1, shape=[num_filters]), trainable=False)  # initialize the filters' biases
                # The conv2d operation expects a 4-D tensor with dimensions corresponding to batch, width, height and channel.
                # The result of our embedding doesn’t contain the channel dimension
                # So we add it manually, leaving us with a layer of shape [None, sequence_length, embedding_size, 1].
                x_embed_expanded = tf.expand_dims(x_embed, -1)
                conv = tf.nn.conv2d(x_embed_expanded, W_filter, strides=[1, 1, 1, 1], padding="VALID")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b_filter), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.params['max_domain_segments_len'] - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID')
                pooled_outputs.append(pooled)
            # Combine all the pooled features
            h_pool = tf.concat(pooled_outputs, axis=3)
            num_filters_total = num_filters * len(filter_sizes)
            domain_vec_cnn2 = tf.reshape(h_pool, [-1, num_filters_total])
            domain_vec_cnn2 = tf.layers.dropout(domain_vec_cnn2, dropout_rate, training=False)
            domain_vectors.append(domain_vec_cnn2)



            # concatenate suffix one-hot and the abstract representation of the domains segments
            # The shape of cat_layer should be [batch_size, n_lstm_neurons+self.params['num_suffix']]
            cat_layer = tf.concat(domain_vectors + [x_suffix], -1)
            # print(cat_layer.get_shape())

            logits = cat_layer
            for _ in range(n_fc_layers):
                logits = tf.contrib.layers.fully_connected(logits, num_outputs=n_rnn_neurons, activation_fn=act_fn, trainable=False)
                logits = tf.layers.dropout(logits, dropout_rate, training=False)

            logits2 = tf.contrib.layers.fully_connected(logits, self.params['num_targets'], activation_fn=act_fn, trainable=False)

        saver_domain = tf.train.Saver([v for v in tf.all_variables() if 'domain_cnn' in v.name])



        with tf.variable_scope('combine_weighting'):
            pooled_outputs = []
            for filter_size in filter_sizes:
                # Define and initialize filters
                filter_shape = [filter_size, embed_dimen_weighting, 1, num_filter_weighting]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1)) # initialize the filters' weights
                b_filter = tf.Variable(tf.constant(0.1, shape=[num_filter_weighting]))  # initialize the filters' biases
                # The conv2d operation expects a 4-D tensor with dimensions corresponding to batch, width, height and channel.
                # The result of our embedding doesn’t contain the channel dimension
                # So we add it manually, leaving us with a layer of shape [None, sequence_length, embedding_size, 1].
                x_embed_expanded = tf.expand_dims(x_embed_weighting, -1)
                conv = tf.nn.conv2d(x_embed_expanded, W_filter, strides=[1, 1, 1, 1], padding="VALID")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b_filter), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.params['max_domain_segments_len'] - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID')
                pooled_outputs.append(pooled)
            # Combine all the pooled features
            h_pool = tf.concat(pooled_outputs, axis=3)
            num_filters_total = num_filter_weighting * len(filter_sizes)
            domain_vec_cnn3 = tf.reshape(h_pool, [-1, num_filters_total])
            domain_vec_cnn3 = tf.nn.l2_normalize(domain_vec_cnn3, dim=-1)

        logits_weight = domain_vec_cnn3
        for _ in range(n_fc_layers):
            logits_weight = tf.contrib.layers.fully_connected(logits_weight, num_outputs=num_fclayer_width_weighting, activation_fn=act_fn)
            # logits_weight = tf.layers.dropout(logits_weight, dropout_rate)

        # logits_weight = tf.concat([logits1, logits2], -1)

        imp = tf.contrib.layers.fully_connected(logits_weight, 2, activation_fn=tf.nn.softmax)
        # domain_imp = domain_imp.transpose()


        desc_imp, domain_imp = tf.unstack(imp, axis=1)
        logits_combine = tf.reshape(desc_imp, [-1,1]) * tf.nn.softmax(logits1) + tf.reshape(domain_imp, [-1,1]) * tf.nn.softmax(logits2)
        # logits_combine = tf.reshape(desc_imp, [-1, 1]) * logits1 + tf.reshape(domain_imp, [-1, 1]) * logits2
        # domain_imp = tf.constant(0.99999)
        # desc_imp = tf.Variable(tf.constant(0.0001))
        #
        # logits_combine = tf.add(tf.multiply(domain_imp, tf.nn.softmax(logits2)),
        #                         tf.multiply(tf.subtract(tf.constant(1.0), tf.divide(desc_imp, desc_imp)), tf.nn.softmax(logits1)))

        # logits_combine = tf.reduce_sum(domain_imp * tf.stack([tf.nn.softmax(logits1), tf.nn.softmax(logits2)]), 1)

        crossentropy = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(y, self.params['num_targets']) * tf.log(logits_combine), [1]))


        # logits_combine = tf.concat([logits1, logits2], -1)
        # logits_combine = tf.contrib.layers.fully_connected(logits_combine,
        #                                                    self.params['num_targets'],
        #                                                    activation_fn=tf.nn.relu)

        # crossentropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits_combine)


        loss_mean = tf.reduce_mean(crossentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate)
        training_op = optimizer.minimize(loss_mean)

        prediction = tf.argmax(logits_combine, axis=-1)
        is_correct = tf.nn.in_top_k(logits_combine, y, 1) # logits are unscaled, but here we only care the argmax
        n_correct = tf.reduce_sum(tf.cast(is_correct, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            init.run()

            saver_desc.restore(sess, os.path.join(OUTPUT_DIR, 'domain_mapping.params'))
            saver_domain.restore(sess, os.path.join(OUTPUT_DIR, 'domain_classifying.params'))
            print("Model restored.")

            n_total_batches = int(np.ceil(len(self.domains_train) / batch_size))
            test_fscore_history = []
            for epoch in range(1, n_epochs + 1):
                # model training
                n_batch = 0
                for X_batch_embed, X_batch_weighting, domain_actual_lens, X_batch_suf, y_batch in self.next_batch(self.domains_train):
                    _, acc_batch_train, loss_batch_train, prediction_train = sess.run([training_op, accuracy, loss_mean, prediction],
                                                                    feed_dict={
                                                                               'bool_train:0': True,
                                                                               'embedding:0': X_batch_embed,
                                                                               'indices:0': X_batch_weighting,
                                                                               'suffix:0': X_batch_suf,
                                                                               'length:0': domain_actual_lens,
                                                                               'target:0': y_batch})

                    n_batch += 1
                    if epoch < 2:
                        # print(prediction_train)
                        print("Epoch %d - Batch %d/%d: loss = %.4f, accuracy = %.4f" %
                              (epoch, n_batch, n_total_batches, loss_batch_train, acc_batch_train))


                # evaluation on training data
                eval_nodes = [n_correct, loss_mean, is_correct, prediction,
                              desc_imp, domain_imp, logits1, logits2,
                              # logits1_normal, logits2_normal,
                              # logits1_softmax, logits2_softmax
                              ]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_train, acc_train, _, _ = self.evaluate(self.domains_train, sess, eval_nodes)
                print("*** On Training Set:\tloss = %.6f\taccuracy = %.4f"
                      % (loss_train, acc_train))

                # evaluation on validation data
                loss_val, acc_val, _, _ = self.evaluate(self.domains_val, sess, eval_nodes)
                print("*** On Validation Set:\tloss = %.6f\taccuracy = %.4f"
                      % (loss_val, acc_val))

                # evaluate on test data
                loss_test, acc_test, is_correct_test, pred_test = self.evaluate(self.domains_test, sess, eval_nodes)
                print("*** On Test Set:\tloss = %.6f\taccuracy = %.4f"
                      % (loss_test, acc_test))



                print()
                print("Macro average:")
                precisions_macro, recalls_macro, fscores_macro, _ = precision_recall_fscore_support(
                                              [category2index[domain['categories'][1]] for domain in self.domains_test],
                                               pred_test, average='macro')
                print("Precision (macro): %.4f, Recall (macro): %.4f, F-score (macro): %.4f" %
                      (precisions_macro, recalls_macro, fscores_macro))
                print()



                if not test_fscore_history or fscores_macro > max(test_fscore_history):
                    # the accuracy of this epoch is the largest
                    print("Classification Performance on individual classes:")
                    precisions_none, recalls_none, fscores_none, supports_none = precision_recall_fscore_support(
                        [category2index[domain['categories'][1]] for domain in self.domains_test],
                        pred_test, average=None)
                    print(tabulate(zip((categories[i] for i in range(len(precisions_none))),
                                       precisions_none, recalls_none, fscores_none, supports_none),
                                   headers=['category', 'precision', 'recall', 'f-score', 'support'],
                                   tablefmt='orgtbl'))

                test_fscore_history.append(fscores_macro)




if __name__ == '__main__':
    classifier = PretrainFastTextClassifier()
    classifier.run_graph()