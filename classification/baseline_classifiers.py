'''
Created on Jun 23, 2015

@author: cwang
'''
import re, pickle, numpy
from _collections import defaultdict
from string import punctuation
from random import shuffle
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_recall_fscore_support

DATASET = 'content'  # 'content' or '2340768'

OUTPUT_DIR = '../Output/'

class BaselineClassifier:

    ns = [4, 5, 6, 7, 8]

    def building_vector(self, features):
        vector = defaultdict(int)
        for f in features:
            vector[f] += 1
        return vector

    def features(self, url):
        # print()
        # print(url)

        """ Feature 1: tokens """
        tokens_list = self.tokens(url)
        # print(tokens_list)

        """ Feature 2: n-grams from tokens"""
        ngram_tokens_list = self.ngram_tokens(tokens_list)
        # print(ngram_tokens_list)

        """ Feature 3: n-grams from URL """
        ngram_url_list = self.ngram_url(tokens_list)
        # print(ngram_url_list)

        """ Feature 4: encoding positional information """
        pos_info_list = self.positional_info(tokens_list, ngram_tokens_list)
        # print(pos_info_list)

#         return self.building_vector( tokens_list + ngram_tokens_list +
#                                 ngram_url_list )
        return tokens_list + ngram_tokens_list + ngram_url_list + pos_info_list


    def tokens(self, url):
        raw_tokens = re.split("[" + punctuation + "]+", url.lower())
        clean_tokens = [t for t in raw_tokens if len(t) > 1 and not t.isnumeric() and t != 'http']
        return clean_tokens

    def char_ngram(self, b, n):
        return [b[i:i+n] for i in range(len(b)-n+1)]

    def ngram_tokens(self, tokens):
        ngram_tokens_list = []
        for n in self.ns:
            for t in tokens:
                if len(t) <= n and t not in ngram_tokens_list:
                    ngram_tokens_list.append(t)
                else:
                    ngram_tokens_list.extend(self.char_ngram(t, n))
        return ngram_tokens_list

    def ngram_url(self, tokens):
        t = "".join(tokens)
        ngram_url_list = []
        for n in self.ns:
            ngram_url_list.extend(self.char_ngram(t, n))
        return ngram_url_list

    def positional_info(self, tokens, ngram_tokens):
        pos_ngrams = []
        for i in range(len(tokens)):
            for nt in ngram_tokens:
                if nt in tokens[i]:
                    pos_ngrams.append("_".join([nt, str(i+1)]))
        return pos_ngrams

    def testcases_with_nonzero_vectors(self, X, y):
        '''
        Remove all test cases with all zero vectors
        :param X:
        :param y:
        :return:
        '''
        nonzero_rows, _ = X.nonzero()
        nonzero_indices = numpy.unique(nonzero_rows)
        y_new = numpy.array(y)[nonzero_indices.tolist()]
        return X[nonzero_indices], y_new

    def load_domains(self, filename):
        domains = pickle.load(open(OUTPUT_DIR + filename, 'rb'))
        domains = [d for cat_domains in domains for d in cat_domains]
        return domains

    def get_XY(self, token, desc, filename):
        domains = self.load_domains(filename)
        shuffle(domains)
        X = []
        y = []
        for domain in domains:
            if token == 'char-ngram':
                x = '.'.join([domain['domain'], domain['suffix']])
                if desc:
                    x = ' '.join([x] + domain['tokenized_desc'])
                X.append(' '.join(self.features(x)))
            elif token == 'word':
                x = ' '.join(domain['segmented_domain'])
                if desc:
                    x = ' '.join([x] + domain['tokenized_desc'])
                X.append(x)

            y.append(domain['target'])
        return X, y

    def transform(self, X, y, vectorizer):
        X = vectorizer.transform(X)
        print("X has been transformed")
        # total_size = X.shape[0]
        # X, y = self.testcases_with_nonzero_vectors(X, y)
        # actual_size = X.shape[0]
        # print(actual_size, "out of", total_size, "examples have non-zero feature vectors")
        return X, y


    def buildXY(self, token='word', desc=False):
        hasher = FeatureHasher(n_features=100000)

        print("Loading Training Data...")
        X_train, y_train = self.get_XY(token, desc, 'training_domains_%s.list' % DATASET)

        print("Fitting and transforming X and Y")
        vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5)
        X_train = vectorizer.fit_transform(X_train) # return term-document matrix
        print(X_train.shape[0], "training examples and", X_train.shape[1], "features")


        print("Loading Validation Data...")
        X_val, y_val = self.get_XY(token, False, 'validation_domains_%s.list' % DATASET)
        X_val, y_val = self.transform(X_val, y_val, vectorizer)


        print("Loading Test Data...")
        X_test, y_test = self.get_XY(token, False, 'test_domains_%s.list' % DATASET)
        X_test, y_test = self.transform(X_test, y_test, vectorizer)

        return X_train, y_train, X_val, y_val, X_test, y_test

    '''
    def getCVIndex(self, y):
        print("================================= Cross-Validation =========================================")
        shuffle = ShuffleSplit(len(y), n_iter=3, test_size=0.2)
        n=0
        for _, test_index in shuffle:
            n+=1
            print("Test size of each class in cv", n)
            for c in range(len(numpy.unique(y))):
                print("Class", c, ":", list(y[test_index]).count(c))
            print()
        print()
        return shuffle

    def getCVResult(self, shuffle):
        output_mode = 'macro'
#         output_mode = None
        default_val = numpy.zeros(len(self.dmoz.selected_layer1)) if output_mode == None else 0.0

        clf = LinearSVC(C=0.001, penalty='l2', verbose=0, class_weight='auto')
#         clf = SVC(C=0.001, verbose=True, class_weight='auto', kernal='poly')
        print("Doing cross-validation")

        cv_res = {}
        for cv_num, (train_index, test_index) in enumerate(shuffle):
            print("*** #CV", cv_num)
            X_train, y_train, X_test, y_test = self.buildXY(train_index, test_index)

            print("Doing cross-validation")
            res = self.get_detailed_evalRes(clf, X_train, y_train, X_test, y_test, output_mode)
            cv_res = self.accumulate_cv_result(cv_res, default_val, float(len(shuffle)), res)
        # Summarize cv results
        cv_res = dict((metric, res / (cv_num + 1)) for metric, res in cv_res.items())
        return cv_res

    def accumulate_cv_result(self, cv_res, default_val, cv_num, this_result):
        precision, recall, fscore, support = this_result
        cv_res['precision'] = cv_res.get('precision', default_val) + precision
        cv_res['recall'] = cv_res.get('recall', default_val) + recall
        cv_res['fscore'] = cv_res.get('fscore', default_val) + fscore
        if support is not None:
            cv_res['support'] = cv_res.get('support', default_val) + support
        return cv_res

    def runCV(self):
        X, y = self.loadModel()
        shuffle = self.getCVIndex(y)

        cv_res = self.getCVResult(X, y, shuffle)

        print("====================================== RESULT ===============================================")
        # print numpy.unique(y)
        print(cv_res)
    '''

    def cal_metrics(self, clf, X, y):
        y_pred = clf.predict(X)
        precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred,
                                                                                 average=None)
        for p, r, f, s in zip(precision, recall, fscore, support):
            print(p, r, f, s)

        precision, recall, fscore, _ = precision_recall_fscore_support(y, y_pred,
                                                                          average='macro')
        print("Precision (macro): %.4f, Recall (macro): %.4f, F-score (macro): %.4f" % (precision, recall, fscore))
        print("Accuracy: %.4f" % accuracy_score(y, y_pred))

    def get_detailed_evalRes(self, clf, X_train, y_train, X_val, y_val, X_test, y_test):

        clf.fit(X_train, y_train)

        # TEST overfitting
        print("\nPerformance on Training Data")
        self.cal_metrics(clf, X_train, y_train)


        print("\nPerformance on Validation Data")
        self.cal_metrics(clf, X_val, y_val)

        print("\nPerformance on Test Data")
        self.cal_metrics(clf, X_test, y_test)



if __name__ == '__main__':
    classifier = BaselineClassifier()
    '''
    token='char-ngram': Baykan2011-based method
    token='word': simple segment-based method
    '''
    X_train, y_train, X_val, y_val, X_test, y_test = classifier.buildXY(token='char-ngram', desc=False)

    clf = LinearSVC(C=0.5, penalty='l2', verbose=0)
    classifier.get_detailed_evalRes(clf, X_train, y_train, X_val, y_val, X_test, y_test)
