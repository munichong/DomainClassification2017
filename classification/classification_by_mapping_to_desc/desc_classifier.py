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

type = 'CNN'

# For CNN
filter_sizes = [2,1]
num_filters = 512

embed_dimen = 300
# n_fc_neurons = 64
dropout_rate= 0.2
n_fc_neurons = 300
n_fc_layers= 3
act_fn = tf.nn.relu

max_required_desc_words_len = 100

n_epochs = 60
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
print("Loading the FastText Model")
# en_model = {"test":np.array([0]*300)}
en_model = FastText.load_fasttext_format('../FastText/wiki.en/wiki.en')


class PretrainFastTextClassifier:

    def __init__(self):
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

        ''' load params '''
        self.params = json.load(open(OUTPUT_DIR + 'params_%s.json' % DATASET))
        self.truncated_desc_words_len = min(max_required_desc_words_len, self.params['max_desc_words_len'])


    def next_batch(self, domains, batch_size=batch_size):
        X_batch_embed = []
        X_batch_suf = []
        y_batch = []
        shuffle(domains)
        start_index = 0
        while start_index < len(domains):
            for i in range(start_index, min(len(domains), start_index + batch_size)):
                ''' description '''
                # skip if a segment is not in en_model
                embeds = [en_model[w.lower()].tolist() for w in
                          domains[i]['tokenized_desc'][ : self.truncated_desc_words_len] if w in en_model]

                ''' description padding '''
                if len(embeds) < self.truncated_desc_words_len:
                    embeds += [[0] * embed_dimen for _ in range(self.truncated_desc_words_len - len(embeds))]

                X_batch_embed.append(embeds)
                assert len(embeds) == self.truncated_desc_words_len


                one_hot_suf = np.zeros(self.params['num_suffix'])
                one_hot_suf[domains[i]['suffix_indices']] = 1.0 / len(domains[i]['suffix_indices'])
                X_batch_suf.append(one_hot_suf)

                y_batch.append(domains[i]['target'])
            yield np.array(X_batch_embed), np.array(X_batch_suf), np.array(y_batch)

            X_batch_embed.clear()
            X_batch_suf.clear()
            y_batch.clear()
            start_index += batch_size


    def evaluate(self, data, session, eval_nodes):
        total_correct = 0
        total_loss = 0
        total_bool = []
        total_pred = []
        logits = []
        n_batch = 0
        for X_batch_embed, X_batch_suf, y_batch in self.next_batch(data):
            batch_correct, batch_loss, batch_bool, batch_pred, batch_logits = session.run(eval_nodes,
                                                         feed_dict={
                                                                    'bool_train:0': False,
                                                                    'embedding:0': X_batch_embed,
                                                                    'suffix:0': X_batch_suf,
                                                                    'target:0': y_batch})
            total_loss += batch_loss
            total_correct += batch_correct
            total_bool.extend(batch_bool)
            total_pred.extend(batch_pred)
            logits.append(batch_logits)
            n_batch += 1
        return total_loss / n_batch, total_correct / len(data), total_bool, total_pred, logits



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

        y = tf.placeholder(tf.int32, shape=[None], name='target') # Each entry in y must be an index in [0, num_classes)


        desc_vectors = []
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
            desc_vec_cnn = tf.reshape(h_pool, [-1, num_filters_total])
            desc_vec_cnn = tf.layers.dropout(desc_vec_cnn, dropout_rate, training=is_training)
            desc_vectors.append(desc_vec_cnn)


        # concatenate suffix one-hot and the abstract representation of the domains segments
        # The shape of cat_layer should be [batch_size, n_lstm_neurons+self.params['num_suffix']]
        cat_layer = tf.concat(desc_vectors + [x_suffix], -1)
        # print(cat_layer.get_shape())

        logits = cat_layer
        for _ in range(n_fc_layers):
            logits = tf.contrib.layers.fully_connected(logits, num_outputs=n_fc_neurons, activation_fn=act_fn)
            logits = tf.layers.dropout(logits, dropout_rate, training=is_training)

        logits_pred = tf.contrib.layers.fully_connected(logits, self.params['num_targets'], activation_fn=act_fn)


        crossentropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits_pred)

        loss_mean = tf.reduce_mean(crossentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate)
        training_op = optimizer.minimize(loss_mean)

        prediction = tf.argmax(logits_pred, axis=-1)
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


        with tf.Session() as sess:
            init.run()
            n_total_batches = int(np.ceil(len(self.domains_train) / batch_size))
            test_fscore_history = []
            for epoch in range(1, n_epochs + 1):
                # model training
                n_batch = 0
                for X_batch_embed, X_batch_suf, y_batch in self.next_batch(self.domains_train):
                    _, acc_batch_train, loss_batch_train, prediction_train = sess.run([training_op, accuracy, loss_mean, prediction],
                                                                    feed_dict={
                                                                               'bool_train:0': True,
                                                                               'embedding:0': X_batch_embed,
                                                                               'suffix:0': X_batch_suf,
                                                                               'target:0': y_batch})

                    n_batch += 1
                    if epoch < 2:
                        # print(prediction_train)
                        print("Epoch %d - Batch %d/%d: loss = %.4f, accuracy = %.4f" %
                              (epoch, n_batch, n_total_batches, loss_batch_train, acc_batch_train))


                # evaluation on training data
                eval_nodes = [n_correct, loss_mean, is_correct, prediction, logits_pred]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_train, acc_train, _, _, logits_train = self.evaluate(self.domains_train, sess, eval_nodes)
                print("*** On Training Set:\tloss = %.6f\taccuracy = %.4f"
                      % (loss_train, acc_train))

                # evaluation on validation data
                loss_val, acc_val, _, _, logits_val = self.evaluate(self.domains_val, sess, eval_nodes)
                print("*** On Validation Set:\tloss = %.6f\taccuracy = %.4f"
                      % (loss_val, acc_val))

                # evaluate on test data
                loss_test, acc_test, is_correct_test, pred_test, logits_test = self.evaluate(self.domains_test, sess, eval_nodes)
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



                    assert len(self.domains_train) == len(logits_train) and \
                           len(self.domains_val) == len(logits_val) and \
                           len(self.domains_test) == len(logits_test)
                    domain2logits = {}
                    # store logits_train, logits_val, and logits_test to the disk
                    for domains, logits_vec in ((self.domains_train, logits_train),
                                                (self.domains_val, logits_val),
                                                (self.domains_test, logits_test)):
                        for domain, vec in zip([domains, logits_vec]):
                            domain2logits[domain['raw_domain']] = vec
                    pickle.dump(domain2logits, open('domain2logits.dict', 'wb'))


                test_fscore_history.append(fscores_macro)




if __name__ == '__main__':
    classifier = PretrainFastTextClassifier()
    classifier.run_graph()