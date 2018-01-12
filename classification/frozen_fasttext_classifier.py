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

char_ngram_sizes=[3,6]

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

n_epochs = 60
batch_size = 4000
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
en_model['wwwreal']

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
        self.max_num_charngrams = sum(self.params['max_segment_char_len'] - size + 1 for size in char_ngram_sizes) + 1
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

    def compute_ngrams(self, word, min_n, max_n):
        BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
        extended_word = BOW + word + EOW
        ngrams = set()
        for i in range(len(extended_word) - min_n + 1):
            for j in range(min_n, max(len(extended_word) - max_n, max_n + 1)):
                ngrams.add(extended_word[i:i + j])
        return ngrams

    def next_batch(self, domains, batch_size=batch_size):
        X_batch_embed = []
        X_batch_suf = []
        domain_actual_lens = []
        sample_weights = []
        y_batch = []
        shuffle(domains)
        start_index = 0
        while start_index < len(domains):
            for i in range(start_index, min(len(domains), start_index + batch_size)):
                # skip if a segment is not in en_model
                '''
                embeds = [en_model[w].tolist() for w in domains[i]['segmented_domain'] if w in en_model]
                '''
                embeds = []
                for w in domains[i]['segmented_domain']:
                    # word_vec = np.copy(en_model.wv.syn0[en_model.wv.vocab[w].index])
                    count = 0
                    # Raises a KeyError since none of the character ngrams of the word are present in the vocabulary
                    try:
                        word_vec = np.copy(en_model[w])
                        count += 1
                    except KeyError:
                        word_vec = np.zeros(embed_dimen)
                    ngrams = self.compute_ngrams(w, char_ngram_sizes[0], char_ngram_sizes[1])
                    for ngram in ngrams:
                        try:
                            word_vec += en_model.wv.syn0_ngrams[en_model.wv.ngrams[ngram]]
                            count += 1
                        except KeyError:
                            continue
                    if count:
                        word_vec /= count
                    embeds.append(word_vec.tolist())


                # ''' padding '''
                # # pad char-ngram level
                # for embed in embeds:
                #     embed + [0] * (self.max_num_charngrams - len(embed))
                #
                # embeds = [embed + [0] * (self.max_num_charngrams - len(embed)) for embed in embeds]
                # embeds += [[0] * self.max_num_charngrams for _ in
                #            range(self.params['max_domain_segments_len'] - len(embeds))]
                # # X_batch_embed.append(tf.pad(embeds, paddings=[[0, n_extra_padding],[0,0]], mode="CONSTANT"))
                # X_batch_embed.append(embeds)
                # print(embeds)
                # print([len(e) for e in embeds])
                # print(np.array(embeds).shape)


                # if not embeds: # Skip if none of segments of this domain can not be recognized by FastText
                #     continue
                domain_actual_lens.append(len(embeds))
                n_extra_padding = self.params['max_domain_segments_len'] - len(embeds)
                embeds += [[0] * embed_dimen for _ in range(n_extra_padding)]
                # X_batch_embed.append(tf.pad(embeds, paddings=[[0, n_extra_padding],[0,0]], mode="CONSTANT"))
                X_batch_embed.append(embeds)
                # print(embeds)
                # print([len(e) for e in embeds])
                # print(np.array(embeds).shape)

                one_hot_suf = np.zeros(self.params['num_suffix'])
                one_hot_suf[domains[i]['suffix_indices']] = 1.0 / len(domains[i]['suffix_indices'])
                X_batch_suf.append(one_hot_suf)

                sample_weights.append(self.class_weights[categories[domains[i]['target']]])
                y_batch.append(domains[i]['target'])

            # print(np.array(X_batch_embed).shape)
            # print(np.array(domain_actual_lens))
            # print(np.array(X_batch_suf))
            # print(np.array(sample_weights))
            # print(np.array(y_batch))

            yield np.array(X_batch_embed), np.array(domain_actual_lens), np.array(X_batch_suf), \
                  np.array(sample_weights), np.array(y_batch)

            # print(sample_weights)

            X_batch_embed.clear()
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
        for X_batch_embed, domain_actual_lens, X_batch_suf, sample_weights, y_batch in self.next_batch(data):
            batch_correct, batch_loss, batch_bool, batch_pred = session.run(eval_nodes,
                                                         feed_dict={
                                                                    'bool_train:0': False,
                                                                    'embedding:0': X_batch_embed,
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
            n_batch += 1
        return total_loss / n_batch, total_correct / len(data), total_bool, total_pred



    def run_graph(self):

        # tf.reset_default_graph()

        # INPUTs
        is_training = tf.placeholder(tf.bool, shape=(), name='bool_train')
        x_embed = tf.placeholder(tf.float32,
                                 shape=[None, self.params['max_domain_segments_len'], embed_dimen],
                                 name='embedding')

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
            rnn_cell = tf.contrib.rnn.BasicRNNCell(n_rnn_neurons, activation=tf.nn.tanh)
            # The shape of last_states should be [batch_size, n_lstm_neurons]
            _, domain_vec_rnn = tf.nn.dynamic_rnn(rnn_cell, x_embed, sequence_length=seq_len, dtype=tf.float32, time_major=False)
            domain_vec_rnn = tf.layers.dropout(domain_vec_rnn, dropout_rate, training=is_training)
            domain_vectors.append(domain_vec_rnn)
        if 'CNN' in type:
            pooled_outputs = []
            for filter_size in filter_sizes:
                # Define and initialize filters
                filter_shape = [filter_size, embed_dimen, 1, num_filters]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1)) # initialize the filters' weights
                b_filter = tf.Variable(tf.constant(0.1, shape=[num_filters]))  # initialize the filters' biases
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
            domain_vec_cnn = tf.reshape(h_pool, [-1, num_filters_total])
            domain_vec_cnn = tf.layers.dropout(domain_vec_cnn, dropout_rate, training=is_training)
            domain_vectors.append(domain_vec_cnn)



        # concatenate suffix one-hot and the abstract representation of the domains segments
        # The shape of cat_layer should be [batch_size, n_lstm_neurons+self.params['num_suffix']]
        cat_layer = tf.concat(domain_vectors + [x_suffix], -1)
        # print(cat_layer.get_shape())

        logits = None
        for _ in range(n_fc_layers):
            logits = tf.contrib.layers.fully_connected(cat_layer, num_outputs=n_rnn_neurons, activation_fn=act_fn)
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

        '''
        # ranking evaluation
        ranked_res = tf.nn.top_k(logits, k=self.params['num_targets'], sorted=True).indices
        y_2d = tf.reshape(y, (tf.shape(y)[0], 1))
        a = tf.where(tf.equal(ranked_res, y_2d))
        a = tf.gather_nd(a, ranked_res)
        print(a)
        print(a.shape)
        rank_sum = tf.reduce_sum(a)[:, -1]
        '''

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
            test_accuracy_history = []
            for epoch in range(1, n_epochs + 1):
                # model training
                n_batch = 0
                for X_batch_embed, domain_actual_lens, X_batch_suf, sample_weights, y_batch in self.next_batch(self.domains_train):
                    _, acc_batch_train, loss_batch_train, prediction_train = sess.run([training_op, accuracy, loss_mean, prediction],
                                                                    feed_dict={
                                                                               'bool_train:0': True,
                                                                               'embedding:0': X_batch_embed,
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
                eval_nodes = [n_correct, loss_mean, is_correct, prediction]
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



                if not test_accuracy_history or acc_test > max(test_accuracy_history):
                    # the accuracy of this epoch is the largest
                    print("Classification Performance on individual classes:")
                    precisions_none, recalls_none, fscores_none, supports_none = precision_recall_fscore_support(
                        [category2index[domain['categories'][1]] for domain in self.domains_test],
                        pred_test, average=None)
                    print(tabulate(zip((categories[i] for i in range(len(precisions_none))),
                                       precisions_none, recalls_none, fscores_none, supports_none),
                                   headers=['category', 'precision', 'recall', 'f-score', 'support'],
                                   tablefmt='orgtbl'))

                    # output all incorrect_prediction
                    with open(os.path.join(OUTPUT_DIR, 'incorrect_predictions.csv'), 'w') as outfile:
                        csv_writer = csv.writer(outfile)
                        csv_writer.writerow(('RAW_DOMAIN', 'SEGMENTED_DOMAIN', 'TRUE_CATEGORY', 'PRED_CATEGORY'))
                        for correct, pred_catIdx, domain in zip(is_correct_test, pred_test, self.domains_test):
                            if correct:
                                continue
                            csv_writer.writerow((domain['raw_domain'],
                                                 domain['segmented_domain'],
                                                 domain['categories'][1],
                                                 categories[pred_catIdx]))
                test_accuracy_history.append(acc_test)




if __name__ == '__main__':
    classifier = PretrainFastTextClassifier()
    classifier.run_graph()