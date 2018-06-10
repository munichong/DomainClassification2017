'''
Created on Oct 30, 2017

@author: munichong
'''
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

DATASET = 'content'  # 'content' or '2340768'

type = 'RNN'
# For RNN
n_rnn_neurons = 300
# For CNN
filter_sizes = [2,1]
num_filters = 512

embed_dimen = 300
# n_fc_neurons = 64
dropout_rate = 0.5
n_fc_layers= 3
act_fn = tf.nn.relu

n_epochs = 50
batch_size = 2000
lr_rate = 0.001

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
en_model = FastText.load_fasttext_format('../FastText/wiki.en/wiki.en')

class PretrainFastTextClassifier:

    def __init__(self):
        ''' load data '''
        self.domains_train = pickle.load(open(OUTPUT_DIR + 'training_domains_%s.list' % DATASET, 'rb'))
        self.domains_train = [d for cat_domains in self.domains_train for d in cat_domains ]
        self.domains_val = pickle.load(open(OUTPUT_DIR + 'validation_domains_%s.list' % DATASET, 'rb'))
        self.domains_val = [d for cat_domains in self.domains_val for d in cat_domains]
        self.domains_test = pickle.load(open(OUTPUT_DIR + 'test_domains_%s.list' % DATASET, 'rb'))
        self.domains_test = [d for cat_domains in self.domains_test for d in cat_domains]

        ''' load params '''
        self.params = json.load(open(OUTPUT_DIR + 'params_%s.json' % DATASET))
        self.compute_class_weights()

    def compute_class_weights(self):
        n_total = sum(self.params['category_dist_traintest'].values())
        n_class = len(self.params['category_dist_traintest'])
        min_w, max_w = 0.5, 1.5
        # min_w, max_w = 0.0, np.inf
        self.class_weights = {cat: max(min(n_total / (n_class * self.params['category_dist_traintest'][cat]), max_w), min_w)
                              for cat, size in self.params['category_dist_traintest'].items()}
        # self.class_weights['Sports'] = 1
        # self.class_weights['Health'] = 1
        # self.class_weights['Business'] = 0.8
        # self.class_weights['Arts'] = 0.8
        pprint(self.class_weights)

    def next_batch(self, domains, batch_size=batch_size):
        X_batch_embed = []
        X_batch_all = []
        X_batch_suf = []
        domain_actual_lens = []
        sample_weights = []
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

                if domains[i]['domain'] in en_model:
                    X_batch_all.append(en_model[domains[i]['domain']])
                else:
                    X_batch_all.append(np.zeros(embed_dimen))

                one_hot_suf = np.zeros(self.params['num_suffix'])
                one_hot_suf[domains[i]['suffix_indices']] = 1.0 / len(domains[i]['suffix_indices'])
                X_batch_suf.append(one_hot_suf)

                sample_weights.append(self.class_weights[categories[domains[i]['target']]])
                y_batch.append(domains[i]['target'])
            yield np.array(X_batch_embed), np.array(X_batch_all), np.array(domain_actual_lens), np.array(X_batch_suf), \
                  np.array(sample_weights), np.array(y_batch)

            # print(sample_weights)

            X_batch_embed.clear()
            X_batch_all.clear()
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
        total_softmax = []
        n_batch = 0
        for X_batch_embed, X_batch_all, domain_actual_lens, X_batch_suf, sample_weights, y_batch in self.next_batch(data):
            batch_correct, batch_loss, batch_bool, batch_pred, batch_softmax = session.run(eval_nodes,
                                                         feed_dict={
                                                                    'bool_train:0': False,
                                                                    'embedding:0': X_batch_embed,
                                                                    'entire_domain:0': X_batch_all,
                                                                    'suffix:0': X_batch_suf,
                                                                    'length:0': domain_actual_lens,
                                                                    'weight:0': sample_weights,
                                                                    'target:0': y_batch})
            # print(batch_bool)
            # print(batch_pred)
            total_loss += batch_loss
            total_correct += batch_correct
            total_bool.extend(batch_bool)
            total_pred.extend(batch_pred)
            total_softmax.extend(batch_softmax)
            n_batch += 1
        return total_loss / n_batch, total_correct / len(data), total_bool, total_pred, total_softmax

    def conv_layer(self, x, filter_size, width):
        conv = tf.layers.conv2d(inputs=x,
                                filters=num_filters,
                                kernel_size=[filter_size, width],
                                strides=[1, width],
                                padding="same",
                                activation=tf.nn.relu,
                                use_bias=True)

        return conv

    def maxpool_layer(self, conv, filter_size):
        maxpool_out = tf.layers.max_pooling2d(inputs=conv,
                                              pool_size=[filter_size, 1],
                                              strides=[1, 1], padding='same')

        return maxpool_out

    def maxpool_layer_last(self, conv):
        maxpool_out = tf.layers.max_pooling2d(inputs=conv,
                                              pool_size=[self.params['max_domain_segments_len'], 1],
                                              strides=[1, 1], padding='valid')
        return maxpool_out

    def run_graph(self):

        # tf.reset_default_graph()

        # INPUTs
        is_training = tf.placeholder(tf.bool, shape=(), name='bool_train')
        x_embed = tf.placeholder(tf.float32,
                                 shape=[None, self.params['max_domain_segments_len'], embed_dimen],
                                 name='embedding')
        x_all = tf.placeholder(tf.float32,
                                 shape=[None, embed_dimen],
                                 name='entire_domain')

        # print(x_embed.get_shape())
        x_suffix = tf.placeholder(tf.float32,
                                  shape=[None, self.params['num_suffix']],
                                  name='suffix')

        seq_len = tf.placeholder(tf.int32, shape=[None], name='length')

        sample_weights = tf.placeholder(tf.float32, shape=[None], name='weight')
        y = tf.placeholder(tf.int32, shape=[None], name='target') # Each entry in y must be an index in [0, num_classes)

        # # embedding layers
        # rnn_input = tf.convert_to_tensor([en_model[w] for w in tf.unstack(x_tokens)], dtype=tf.float32)

        # embeddings = tf.get_variable('embedding_matrix', [len(en_model.vocab), embed_dimen])
        #
        # rnn_input = tf.convert_to_tensor([tf.cond(w in en_model,
        #                                           tf.nn.embedding_lookup(embeddings, x)[0,:],
        #                                           )
        #                                   for w in x_tokens])

        domain_vectors = []
        if 'RNN' in type:
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_rnn_neurons, activation=tf.nn.tanh)
            # The shape of last_states should be [batch_size, n_lstm_neurons]
            _, domain_vec_rnn = tf.nn.dynamic_rnn(rnn_cell, x_embed, sequence_length=seq_len, dtype=tf.float32, time_major=False)
            domain_vec_rnn = tf.layers.dropout(domain_vec_rnn, dropout_rate, training=is_training)
            domain_vectors.append(domain_vec_rnn)
        if 'CNN' in type:
            pooled_outputs = []
            for filter_size in filter_sizes:
                x_embed_expanded = tf.expand_dims(x_embed, -1)

                # print(x_embed_expanded.get_shape())

                conv_out = self.conv_layer(x_embed_expanded, filter_size, embed_dimen)
                # print(conv_out.get_shape())
                maxpool_out = self.maxpool_layer_last(conv_out)
                # print(maxpool_out.get_shape())

                pooled_outputs.append(maxpool_out)

            # Combine all the pooled features
            h_pool = tf.concat(pooled_outputs, axis=3)
            num_filters_total = num_filters * len(filter_sizes)
            domain_vec_cnn = tf.reshape(h_pool, [-1, num_filters_total])

            # filter_weights = tf.Variable(tf.truncated_normal([num_filters_total], stddev=0.1))
            # domain_vec_cnn = domain_vec_cnn * filter_weights

            domain_vec_cnn = tf.layers.dropout(domain_vec_cnn, dropout_rate, training=is_training)
            domain_vectors.append(domain_vec_cnn)


        # concatenate suffix one-hot and the abstract representation of the domains segments
        # The shape of cat_layer should be [batch_size, n_lstm_neurons+self.params['num_suffix']]
        cat_layer = tf.concat(domain_vectors + [x_suffix, x_all], -1)

        logits = cat_layer
        for _ in range(n_fc_layers):
            logits = tf.layers.dense(logits, units=n_rnn_neurons, activation=act_fn)
            logits = tf.layers.dropout(logits, dropout_rate, training=is_training)

        logits = tf.layers.dense(logits, units=self.params['num_targets'], activation=act_fn)


        if class_weighted:
            crossentropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, weights=sample_weights)
        else:
            crossentropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

        loss_mean = tf.reduce_mean(crossentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate)
        training_op = optimizer.minimize(loss_mean)

        prediction = tf.argmax(logits, axis=-1)
        prediction_softmax = tf.nn.softmax(logits)
        is_correct = tf.nn.in_top_k(logits, y, 1) # logits are unscaled, but here we only care the argmax
        n_correct = tf.reduce_sum(tf.cast(is_correct, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


        init = tf.global_variables_initializer()


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
            n_total_batches = int(np.ceil(len(self.domains_train) / batch_size))
            max_val_fscore = -1
            for epoch in range(1, n_epochs + 1):
                # model training
                n_batch = 0
                for X_batch_embed, X_batch_all, domain_actual_lens, X_batch_suf, sample_weights, y_batch in self.next_batch(self.domains_train):
                    _, acc_batch_train, loss_batch_train, prediction_train = sess.run([training_op, accuracy, loss_mean, prediction],
                                                                    feed_dict={
                                                                               'bool_train:0': True,
                                                                               'embedding:0': X_batch_embed,
                                                                               'entire_domain:0': X_batch_all,
                                                                               'suffix:0': X_batch_suf,
                                                                               'length:0': domain_actual_lens,
                                                                               'weight:0': sample_weights,
                                                                               'target:0': y_batch})

                    n_batch += 1
                    if epoch < 2:
                        # print(prediction_train)
                        print("Epoch %d - Batch %d/%d: loss = %.4f, accuracy = %.4f" %
                              (epoch, n_batch, n_total_batches, loss_batch_train, acc_batch_train))


                # evaluation on training data
                eval_nodes = [n_correct, loss_mean, is_correct, prediction, prediction_softmax]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_train, acc_train, _, _, _ = self.evaluate(self.domains_train, sess, eval_nodes)
                print("*** On Training Set:\tloss = %.6f\taccuracy = %.4f"
                      % (loss_train, acc_train))

                # evaluation on validation data
                loss_val, acc_val, is_correct_val, pred_val, softmax_val = self.evaluate(self.domains_val, sess, eval_nodes)
                print("*** On Validation Set:\tloss = %.6f\taccuracy = %.4f"
                      % (loss_val, acc_val))

                # evaluate on test data
                loss_test, acc_test, is_correct_test, pred_test, softmax_test = self.evaluate(self.domains_test, sess, eval_nodes)
                print("*** On Test Set:\tloss = %.6f\taccuracy = %.4f"
                      % (loss_test, acc_test))



                precisions_macro_val, recalls_macro_val, fscores_macro_val, _ = precision_recall_fscore_support(
                    [category2index[domain['categories'][1]] for domain in self.domains_val],
                    pred_val, average='macro')
                print("Precision (macro): %.4f, Recall (macro): %.4f, F-score (macro): %.4f" %
                      (precisions_macro_val, recalls_macro_val, fscores_macro_val))



                if max_val_fscore < 0 or fscores_macro_val > max_val_fscore:
                    max_val_fscore = fscores_macro_val
                    print("GET THE HIGHEST ***F-SCORE*** ON THE ***VALIDATION DATA***!")
                    print()



                    print("[*** Test Data ***] Classification Performance on individual classes:")
                    precisions_none, recalls_none, fscores_none, supports_none = precision_recall_fscore_support(
                        [category2index[domain['categories'][1]] for domain in self.domains_test], pred_test,
                        average=None)
                    print(tabulate(zip((categories[i] for i in range(len(precisions_none))),
                                       precisions_none, recalls_none, fscores_none, supports_none),
                                   headers=['category', 'precision', 'recall', 'f-score', 'support'],
                                   tablefmt='orgtbl'))
                    precisions_macro_test, recalls_macro_test, fscores_macro_test, _ = precision_recall_fscore_support(
                        [category2index[domain['categories'][1]] for domain in self.domains_test],
                        pred_test, average='macro')

                    mrr, avg_r = self.mean_reciprocal_rank(self.domains_test, softmax_test)
                    print("Precision (macro): %.4f, Recall (macro): %.4f, F-score (macro): %.4f, "
                          "\nMean Reciprocal Rank: %.4f, Average Rank: %.4f" %
                          (precisions_macro_test, recalls_macro_test, fscores_macro_test, mrr, avg_r))



                    # output all incorrect_prediction on VALIDATION data
                    with open(os.path.join(OUTPUT_DIR, 'incorrect_predictions.csv'), 'w', newline="\n") as outfile:
                        csv_writer = csv.writer(outfile)
                        csv_writer.writerow(('RAW_DOMAIN', 'SEGMENTED_DOMAIN', 'TRUE_CATEGORY', 'PRED_CATEGORY'))
                        for correct, pred_catIdx, domain in zip(is_correct_val, pred_val, self.domains_val):
                            if correct:
                                continue
                            csv_writer.writerow((domain['raw_domain'],
                                                 domain['segmented_domain'],
                                                 domain['categories'][1],
                                                 categories[pred_catIdx]))




                    # output all prediction on VALIDATION data
                    with open(os.path.join(OUTPUT_DIR, 'all_predictions_frozen.csv'), 'w', newline="\n") as outfile:
                        csv_writer = csv.writer(outfile)
                        csv_writer.writerow(sorted(category2index.items(), key=lambda x: x[1]))
                        csv_writer.writerow(('RAW_DOMAIN', 'SEGMENTED_DOMAIN', 'TRUE_CATEGORY', 'PRED_CATEGORY'))
                        for correct, pred_catIdx, domain, pred_softmax in zip(is_correct_val, pred_val,
                                                                              self.domains_val, softmax_val):
                            csv_writer.writerow((domain['raw_domain'],
                                                 domain['segmented_domain'],
                                                 domain['categories'][1],
                                                 categories[pred_catIdx],
                                                 str(pred_softmax)))




    def mean_reciprocal_rank(self, domains, softmax_pred):
        reciprocal_rank = 0
        average_rank = 0
        for domain, sm_pred in zip(domains, softmax_pred):
            catIdx_true = category2index[domain['categories'][1]]
            sorted_res = sorted(enumerate(sm_pred), key=lambda x: x[1], reverse=True)
            rank = [rank for rank, (catIdx, score) in enumerate(sorted_res, start=1) if catIdx == catIdx_true][0]
            reciprocal_rank += 1.0 / rank
            average_rank += rank
        return reciprocal_rank / len(domains), average_rank / len(domains)





if __name__ == '__main__':
    classifier = PretrainFastTextClassifier()
    classifier.run_graph()