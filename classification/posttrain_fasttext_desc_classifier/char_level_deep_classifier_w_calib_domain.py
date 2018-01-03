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
from tensorflow.contrib import slim
import domain_desc_calibration as calib

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.wrappers import FastText

DATASET = 'content'  # 'content' or '2340768'

char_ngram = 4

domain_network_type_indep = 'CNN'
domain_network_type_calib = 'CNN'
# For RNN
n_rnn_neurons = 300
# For CNN
domain_filter_sizes_indep = [2,1]
domain_num_filters_indep = 256

domain_filter_sizes_calib = calib.domain_filter_sizes
domain_num_filters_calib = calib.domain_num_filters

char_embed_dimen_indep = 50
char_embed_dimen_calib = calib.char_embed_dimen

dropout_rate = 0.2
dropout_rate_indep= 0.2
dropout_rate_calib = calib.dropout_rate
n_fc_layers_domain_indep= 3
width_fc_layers_domain_indep = 300
n_fc_layers_calib = calib.n_fc_layers_domain
width_fc_layers_calib = calib.width_fc_layers_domain
width_final_rep = calib.width_final_rep
n_fc_layers = 1
width_fc_layers = 300

act_fn = tf.nn.relu

truncated_desc_words_len = calib.truncated_desc_words_len

n_epochs = 50
batch_size = 2000
lr_rate = 0.001

class_weighted = False


OUTPUT_DIR = '../../Output/'

category2index = pickle.load(open(os.path.join(OUTPUT_DIR + 'category2index_%s.dict' % DATASET), 'rb'))
categories = [''] * len(category2index)
for cate, i in category2index.items():
    categories[i] = cate
print(categories)

# Creating the model
# print("Loading the FastText Model")
# en_model = {"test":np.array([0]*300)}
# en_model = FastText.load_fasttext_format('../FastText/wiki.en/wiki.en')



class CharLevelClassifier_w_calib_domain:

    def __init__(self):
        ''' load data '''
        # origin_train_domains = pickle.load(open(OUTPUT_DIR + 'training_domains_%s.list' % DATASET, 'rb'))
        # self.domains_train = []
        # self.domains_val = []
        # self.domains_test = []
        # for domains in origin_train_domains:
        #     shuffle(domains)
        #     train_end_index = int(len(domains) * 0.8)
        #     self.domains_train.append(domains[ : train_end_index])
        #     validation_end_index = int(len(domains) * (0.8 + (1 - 0.8) / 2))
        #     self.domains_val.append(domains[train_end_index : validation_end_index] )
        #     self.domains_test.append(domains[validation_end_index: ])
        #
        # # self.domains_train = pickle.load(open(OUTPUT_DIR + 'training_domains_%s.list' % DATASET, 'rb'))
        # self.domains_train = [d for cat_domains in self.domains_train for d in cat_domains]
        # # self.domains_val = pickle.load(open(OUTPUT_DIR + 'validation_domains_%s.list' % DATASET, 'rb'))
        # self.domains_val = [d for cat_domains in self.domains_val for d in cat_domains]
        # # self.domains_test = pickle.load(open(OUTPUT_DIR + 'test_domains_%s.list' % DATASET, 'rb'))
        # self.domains_test = [d for cat_domains in self.domains_test for d in cat_domains]
        # print(len(self.domains_train), len(self.domains_val), len(self.domains_test))

        self.domains_train = pickle.load(open(OUTPUT_DIR + 'training_domains_%s.list' % DATASET, 'rb'))
        self.domains_train = [d for cat_domains in self.domains_train for d in cat_domains]
        self.domains_val = pickle.load(open(OUTPUT_DIR + 'validation_domains_%s.list' % DATASET, 'rb'))
        self.domains_val = [d for cat_domains in self.domains_val for d in cat_domains]
        self.domains_test = pickle.load(open(OUTPUT_DIR + 'test_domains_%s.list' % DATASET, 'rb'))
        self.domains_test = [d for cat_domains in self.domains_test for d in cat_domains]

        ''' convert char n-gram and words to indice, respectively '''
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
        X_batch_embed = []
        X_batch_embed_mask = []
        X_batch_suf = []
        domain_actual_lens = []
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
                X_batch_embed.append(embeds)
                ''' domain char n-gram mask '''
                X_batch_embed_mask.append((np.array(embeds) != 0).astype(float))

                ''' top-level domain (suffix) '''
                one_hot_suf = np.zeros(self.params['num_suffix'])
                one_hot_suf[domains[i]['suffix_indices']] = 1.0 / len(domains[i]['suffix_indices'])
                X_batch_suf.append(one_hot_suf)

                ''' target category '''
                sample_weights.append(self.class_weights[categories[domains[i]['target']]])
                y_batch.append(domains[i]['target'])

            yield np.array(X_batch_embed), np.array(X_batch_embed_mask), np.array(domain_actual_lens), \
                  np.array(X_batch_suf), np.array(sample_weights), np.array(y_batch)

            # print(sample_weights)


            X_batch_embed.clear()
            X_batch_embed_mask.clear()
            domain_actual_lens.clear()
            X_batch_suf.clear()
            sample_weights.clear()
            y_batch.clear()
            start_index += batch_size


    def evaluate(self, data, session, eval_nodes):
        total_correct = 0
        total_loss = 0
        total_bool = []
        total_pred = []
        n_batch = 0
        for X_batch_embed, X_batch_mask, domain_actual_lens, X_batch_suf, sample_weights, y_batch in self.next_batch(data):
            batch_correct, batch_loss, batch_bool, batch_pred = session.run(eval_nodes,
                                                         feed_dict={
                                                                    'bool_train:0': False,
                                                                    'domain_embed:0': X_batch_embed,
                                                                    'domain_mask:0': X_batch_mask,
                                                                    'suffix:0': X_batch_suf,
                                                                    'length:0': domain_actual_lens,
                                                                    'weight:0': sample_weights,
                                                                    'target:0': y_batch})
            total_loss += batch_loss
            total_correct += batch_correct
            total_bool.extend(batch_bool)
            total_pred.extend(batch_pred)
            n_batch += 1
        return total_loss / n_batch, total_correct / n_batch, total_bool, total_pred


    def get_cnn_output(self, embed_dimen, embed, max_seq_len, num_filters, filter_sizes, is_training, cnn_dropout, trainable=True):
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
        cnn_output = tf.layers.dropout(cnn_output, cnn_dropout, training=is_training)
        return cnn_output


    def run_graph(self):

        # tf.reset_default_graph()

        # INPUTs
        is_training = tf.placeholder(tf.bool, shape=(), name='bool_train')
        x_suffix = tf.placeholder(tf.float32,
                                  shape=[None, self.params['num_suffix']],
                                  name='suffix')


        x_char_ngram_indice = tf.placeholder(tf.int32,
                                 shape=[None, self.params['max_domain_segments_len'],
                                        self.params['max_segment_char_len'] - char_ngram + 1],
                                 name='domain_embed')



        seq_len = tf.placeholder(tf.int32, shape=[None], name='length')

        domain_mask = tf.placeholder(tf.float32, shape=[None, self.params['max_domain_segments_len'],
                                                 self.params['max_segment_char_len'] - char_ngram + 1],
                              name='domain_mask')



        output_vectors = []

        ''' independent domain part '''
        domain_embeddings_indep = tf.Variable(
            tf.random_uniform([len(self.charngram2index), char_embed_dimen_indep], -1.0, 1.0))
        domain_embed_indep = tf.nn.embedding_lookup(domain_embeddings_indep, x_char_ngram_indice)
        domain_mask_indep = tf.expand_dims(domain_mask, axis=-1)
        domain_mask_indep = tf.tile(domain_mask_indep, [1, 1, 1, char_embed_dimen_indep])
        domain_embed_indep = tf.multiply(domain_embed_indep,
                                     domain_mask_indep)  # x_embed_domain.shape: (None, self.params['max_domain_segments_len'],
        # self.params['max_segment_char_len'] - char_ngram + 1, char_embed_dimen)
        domain_embed_indep = tf.reduce_mean(domain_embed_indep, 2)


        # if 'RNN' in domain_network_type:
        #     rnn_cell = tf.contrib.rnn.BasicRNNCell(n_rnn_neurons, activation=tf.nn.tanh)
        #     # The shape of last_states should be [batch_size, n_lstm_neurons]
        #     _, domain_vec_rnn = tf.nn.dynamic_rnn(rnn_cell, x_embed_domain, sequence_length=seq_len, dtype=tf.float32, time_major=False)
        #     domain_vec_rnn = tf.layers.dropout(domain_vec_rnn, dropout_rate, training=is_training)
        #
        #     for _ in range(n_fc_layers_domain):
        #         logits_domain_rnn = tf.contrib.layers.fully_connected(domain_vec_rnn, num_outputs=width_fc_layers_domain,
        #                                                           activation_fn=act_fn)
        #         logits_domain_rnn = tf.layers.dropout(logits_domain_rnn, dropout_rate, training=is_training)
        #     output_vectors.append(logits_domain_rnn)

        if 'CNN' in domain_network_type_indep:
            with tf.variable_scope('cnn_domain_indep'):

                domain_vec_cnn = self.get_cnn_output(char_embed_dimen_indep, domain_embed_indep,
                                                     self.params['max_domain_segments_len'], domain_num_filters_indep,
                                                     domain_filter_sizes_indep, is_training, dropout_rate_indep)

                for _ in range(n_fc_layers_domain_indep):
                    logits_domain_cnn = tf.contrib.layers.fully_connected(domain_vec_cnn, num_outputs=width_fc_layers_domain_indep,
                                                                          activation_fn=act_fn)
                    logits_domain_cnn = tf.layers.dropout(logits_domain_cnn, dropout_rate_indep, training=is_training)
            output_vectors.append(logits_domain_cnn)




        ''' calibrated domain part '''
        domain_embeddings_calib = tf.Variable(
            tf.random_uniform([len(self.charngram2index), char_embed_dimen_calib], -1.0, 1.0),
            trainable=False)
        domain_embed_calib = tf.nn.embedding_lookup(domain_embeddings_calib, x_char_ngram_indice)
        domain_mask_calib = tf.expand_dims(domain_mask, axis=-1)
        domain_mask_calib = tf.tile(domain_mask_calib, [1, 1, 1, char_embed_dimen_calib])
        domain_embed_calib = tf.multiply(domain_embed_calib, domain_mask_calib)
        domain_embed_calib = tf.reduce_mean(domain_embed_calib, 2)

        sample_weights = tf.placeholder(tf.float32, shape=[None], name='weight')
        y = tf.placeholder(tf.int32, shape=[None],
                           name='target')  # Each entry in y must be an index in [0, num_classes)

        if 'CNN' in domain_network_type_calib:
            with tf.variable_scope('cnn_domain_calib'):

                domain_vec_cnn = self.get_cnn_output(char_embed_dimen_calib, domain_embed_calib,
                                                     self.params['max_domain_segments_len'], domain_num_filters_calib,
                                                     domain_filter_sizes_calib, is_training, dropout_rate_calib, trainable=False)

                for _ in range(n_fc_layers_calib - 1):
                    logits_domain = tf.contrib.layers.fully_connected(domain_vec_cnn, num_outputs=width_fc_layers_calib,
                                                                    activation_fn=act_fn, trainable=False)
                    logits_domain = tf.layers.dropout(logits_domain, dropout_rate_calib, training=is_training)

                logits_domain = tf.contrib.layers.fully_connected(logits_domain, num_outputs=width_final_rep,
                                                                  activation_fn=act_fn, trainable=False)
                logits_domain = tf.layers.dropout(logits_domain, dropout_rate, training=is_training)

            output_vectors.append(logits_domain)

        cat_layer = tf.concat(output_vectors + [x_suffix], -1)


        for _ in range(n_fc_layers):
            logits = tf.contrib.layers.fully_connected(cat_layer, num_outputs=width_fc_layers, activation_fn=act_fn)
            logits = tf.layers.dropout(logits, dropout_rate, training=is_training)

        logits = tf.contrib.layers.fully_connected(logits, self.params['num_targets'], activation_fn=act_fn)

        if class_weighted:
            crossentropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, weights=sample_weights)
        else:
            crossentropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

        loss_mean = tf.reduce_mean(crossentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate)
        training_op = optimizer.minimize(loss_mean)

        prediction = tf.argmax(logits, axis=-1)
        is_correct = tf.nn.in_top_k(logits, y, 1) # logits are unscaled, but here we only care the argmax
        n_correct = tf.reduce_sum(tf.cast(is_correct, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        init = tf.global_variables_initializer()

        print('Trainable Variables:')
        print(tf.trainable_variables())

        variables = slim.get_variables_to_restore()
        variables_to_restore = {v.name : v for v in variables if v.name.split('/')[0] == 'cnn_domain_calib'}
        print("variables_to_restore:", ['domain_embeddings_calib'] + sorted(variables_to_restore.keys()))
        saver_for_calib_restore = tf.train.Saver({**{"domain_embeddings_calib": domain_embeddings_calib}, **variables_to_restore})

        ''' Make sure all variables about the domain calibration part are non-trainable '''
        assert [] == [v for v in tf.trainable_variables() if v.name.split('/')[0] == 'cnn_domain_calib']


        ''' For TensorBoard '''
        '''
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        root_log_dir = 'tf_logs'
        logdir = OUTPUT_DIR + root_log_dir + "/run-" + now + "/"
        loss_summary = tf.summary.scalar('loss_mean', loss_mean)
        acc_summary = tf.summary.scalar('accuracy', accuracy)
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        '''

        with tf.Session() as sess:
            init.run()

            # Restore variables from disk.
            saver_for_calib_restore.restore(sess, os.path.join(OUTPUT_DIR, 'domain_abstraction.params'))
            print("Model restored.")


            n_total_batches = int(np.ceil(len(self.domains_train) / batch_size))

            test_macro_fscore_history = []
            for epoch in range(1, n_epochs + 1):
                # model training
                n_batch = 0
                for X_batch_embed, X_batch_mask, domain_actual_lens, X_batch_suf, sample_weights, y_batch in self.next_batch(self.domains_train):
                    _, acc_batch_train, loss_batch_train, prediction_train = sess.run([training_op, accuracy, loss_mean, prediction],
                                                                    feed_dict={
                                                                               'bool_train:0': True,
                                                                               'domain_embed:0': X_batch_embed,
                                                                               'domain_mask:0': X_batch_mask,
                                                                               'suffix:0': X_batch_suf,
                                                                               'length:0': domain_actual_lens,
                                                                               'weight:0': sample_weights,
                                                                               'target:0': y_batch})

                    n_batch += 1
                    if epoch < 2:
                        # print(prediction_train)
                        print("Epoch %d - Batch %d/%d: loss = %.4f, accuracy = %.4f" %
                              (epoch, n_batch, n_total_batches, loss_batch_train, acc_batch_train))


                ''''''''''''''''''''''''''''''''''''
                ''' evaluation on training data '''
                ''''''''''''''''''''''''''''''''''''
                eval_nodes = [n_correct, loss_mean, is_correct, prediction]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_train, acc_train, _, _ = self.evaluate(self.domains_train, sess, eval_nodes)
                print("*** On Training Set:\tloss = %.6f\taccuracy = %.4f" % (loss_train, acc_train))


                ''''''''''''''''''''''''''''''''''''''
                ''' evaluation on validation data '''
                ''''''''''''''''''''''''''''''''''''''
                loss_val, acc_val, _, pred_val = self.evaluate(self.domains_val, sess, eval_nodes)
                print("*** On Validation Set:\tloss = %.6f\taccuracy = %.4f" % (loss_val, acc_val))


                ''''''''''''''''''''''''''''''
                ''' evaluate on test data '''
                ''''''''''''''''''''''''''''''
                loss_test, acc_test, is_correct_test, pred_test = self.evaluate(self.domains_test, sess, eval_nodes)
                print("*** On Test Set:\tloss = %.6f\taccuracy = %.4f" % (loss_test, acc_test))

                print()
                print("Macro average:")
                precisions_macro, recalls_macro, fscores_macro, _ = precision_recall_fscore_support(
                    [category2index[domain['categories'][1]] for domain in self.domains_test],
                    pred_test, average='macro')
                print("Precision (macro): %.4f, Recall (macro): %.4f, F-score (macro): %.4f" %
                      (precisions_macro, recalls_macro, fscores_macro))
                print()

                test_macro_fscore_history.append(fscores_macro)

                if fscores_macro >= max(test_macro_fscore_history):
                    print("Classification Performance on individual classes <Validation Data>:")
                    precisions_none, recalls_none, fscores_none, supports_none = precision_recall_fscore_support(
                        [category2index[domain['categories'][1]] for domain in self.domains_test],
                        pred_test, average=None)
                    print(tabulate(zip((categories[i] for i in range(len(precisions_none))),
                                       precisions_none, recalls_none, fscores_none, supports_none),
                                   headers=['category', 'precision', 'recall', 'f-score', 'support'],
                                   tablefmt='orgtbl'))





if __name__ == '__main__':
    classifier = CharLevelClassifier_w_calib_domain()
    classifier.run_graph()
