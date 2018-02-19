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
# For RNN
n_rnn_neurons = 512
# For CNN
filter_sizes = [2,1]

num_filters = 512

embed_dimen = 300
# n_fc_neurons = 64
dropout_rate= 0.2
n_fc_layers= 3
act_fn = tf.nn.relu

n_epochs = 30
batch_size = 32
lr_rate = 0.001

calibration_reg_factor = 0.00  # regularization leads to worse performance
autoencoder_lr_rate = 0.001


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


domain2logits = pickle.load(open('domain2logits_logits_pred_relu.dict', 'rb'))


class DescDomainMapper:

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


        self.domains_train = [d for cat_domains in self.domains_train for d in cat_domains]
        self.domains_val = [d for cat_domains in self.domains_val for d in cat_domains]
        self.domains_test = [d for cat_domains in self.domains_test for d in cat_domains]
        print(len(self.domains_train), len(self.domains_val), len(self.domains_test))

        ''' load params '''
        self.params = json.load(open(OUTPUT_DIR + 'params_%s.json' % DATASET))


    def next_batch(self, domains, batch_size=batch_size):
        X_batch_domain = []
        X_batch_desc = []
        shuffle(domains)
        start_index = 0
        while start_index < len(domains):
            for i in range(start_index, min(len(domains), start_index + batch_size)):
                ''' domain '''
                # skip if a segment is not in en_model
                embeds_domain = [en_model[w].tolist() for w in domains[i]['segmented_domain'] if w in en_model]
                # if not embeds_domain: # Skip if none of segments of this domain can not be recognized by FastText
                #     continue

                n_extra_padding = self.params['max_domain_segments_len'] - len(embeds_domain)
                embeds_domain += [[0] * embed_dimen for _ in range(n_extra_padding)]
                # X_batch_embed.append(tf.pad(embeds, paddings=[[0, n_extra_padding],[0,0]], mode="CONSTANT"))
                X_batch_domain.append(embeds_domain)

                ''' description '''
                embeds_desc = domain2logits[domains[i]['raw_domain']]
                X_batch_desc.append(embeds_desc)


            yield np.array(X_batch_domain), np.array(X_batch_desc)

            X_batch_domain.clear()
            X_batch_desc.clear()
            start_index += batch_size


    def evaluate(self, data, session, eval_nodes):
        total_loss = 0
        total_mapping = []
        n_batch = 0
        for X_batch_domain, X_batch_desc in self.next_batch(data):
            batch_loss, batch_mapping = session.run(eval_nodes,
                                                         feed_dict={
                                                                    'bool_train:0': False,
                                                                    'domain_embed:0': X_batch_domain,
                                                                    'desc_embed:0': X_batch_desc,
                                                                    })
            total_loss += batch_loss
            total_mapping.extend(batch_mapping)
            n_batch += 1
        return total_loss / n_batch, total_mapping



    def run_graph(self):

        # tf.reset_default_graph()

        # INPUTs
        is_training = tf.placeholder(tf.bool, shape=(), name='bool_train')
        x_domain = tf.placeholder(tf.float32,
                                 shape=[None, self.params['max_domain_segments_len'], embed_dimen],
                                 name='domain_embed')

        # print(x_embed.get_shape())
        x_desc = tf.placeholder(tf.float32,
                                 shape=[None, self.params['num_targets']],
                                 name='desc_embed')

        with tf.variable_scope('domain_mapping'):
            domain_vectors = []
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
                    x_embed_expanded = tf.expand_dims(x_domain, -1)
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

            logits = domain_vectors[0]

            W_T = tf.Variable(tf.truncated_normal([n_rnn_neurons, n_rnn_neurons], stddev=0.1), name="weight_transform")
            b_T = tf.Variable(tf.constant(1.0, shape=[n_rnn_neurons]), name="bias_transform")

            for _ in range(n_fc_layers):
                logits = tf.contrib.layers.fully_connected(logits, num_outputs=n_rnn_neurons, activation_fn=act_fn)
                logits = tf.layers.dropout(logits, dropout_rate, training=is_training)

            T = tf.sigmoid(tf.matmul(logits, W_T) + b_T, name="transform_gate")
            C = tf.subtract(1.0, T, name="carry_gate")

            logits = tf.add(tf.multiply(logits, T), tf.multiply(logits, C), "y")

            logits_pred = tf.contrib.layers.fully_connected(logits, self.params['num_targets'], activation_fn=act_fn)

        reconstruction_loss = tf.sqrt(tf.losses.mean_squared_error(x_desc, logits_pred, weights=1.0))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = reconstruction_loss + calibration_reg_factor * sum(reg_losses)

        optimizer = tf.train.AdamOptimizer(autoencoder_lr_rate)
        training_op = optimizer.minimize(loss)

        # variables_to_save = {v.name: v for v in tf.global_variables()}
        # saver_for_domain_save = tf.train.Saver({**variables_to_save})

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()


        with tf.Session() as sess:
            init.run()
            test_loss_history = []
            n_total_batches = int(np.ceil(len(self.domains_train) / batch_size))
            for epoch in range(1, n_epochs + 1):
                # model training
                n_batch = 0
                for X_batch_domain, X_batch_desc in self.next_batch(self.domains_train):
                    _, loss_batch_train = sess.run([training_op, loss],
                                                                    feed_dict={
                                                                               'bool_train:0': True,
                                                                               'domain_embed:0': X_batch_domain,
                                                                               'desc_embed:0': X_batch_desc,
                                                                               })

                    n_batch += 1
                    if epoch < 2:
                        # print(prediction_train)
                        print("Epoch %d - Batch %d/%d: loss = %.4f" %
                              (epoch, n_batch, n_total_batches, loss_batch_train))

                ''''''''''''''''''''''''''''''''''''
                ''' evaluation on training data '''
                ''''''''''''''''''''''''''''''''''''
                eval_nodes = [loss, logits_pred]
                print()
                print("========== Evaluation at Epoch %d ==========" % epoch)
                loss_train, mapping_train = self.evaluate(self.domains_train, sess, eval_nodes)
                print("*** On Training Set:\tloss = %.6f" % (loss_train))

                ''''''''''''''''''''''''''''''''''''''
                ''' evaluation on validation data '''
                ''''''''''''''''''''''''''''''''''''''
                loss_val, mapping_val = self.evaluate(self.domains_val, sess, eval_nodes)
                print("*** On Validation Set:\tloss = %.6f" % (loss_val))

                ''''''''''''''''''''''''''''''''''''''
                ''' evaluation on test data '''
                ''''''''''''''''''''''''''''''''''''''
                loss_test, mapping_test = self.evaluate(self.domains_test, sess, eval_nodes)
                print("*** On Test Set:\tloss = %.6f" % (loss_test))



                test_loss_history.append(loss_test)
                if loss_test == min(test_loss_history):
                    domain2mapping = {}
                    for domains, mapping_vec in ((self.domains_train, mapping_train),
                                                (self.domains_val, mapping_val),
                                                (self.domains_test, mapping_test)):
                        for domain, vec in zip(domains, mapping_vec):
                            domain2mapping[domain['raw_domain']] = vec
                    pickle.dump(domain2mapping, open('domain2mapping.dict', 'wb'))

                    saver.save(sess, os.path.join(OUTPUT_DIR, 'domain_mapping.params'))
                    print('Save on the disk at %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



if __name__ == '__main__':
    classifier = DescDomainMapper()
    classifier.run_graph()