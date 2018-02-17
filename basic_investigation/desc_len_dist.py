import pickle, numpy
from collections import Counter
from pprint import pprint

DATASET = 'content'  # 'content' or '2340768'
OUTPUT_DIR = '../Output/'


all_len = []
for origin_domains in (pickle.load(open(OUTPUT_DIR + 'training_domains_%s.list' % DATASET, 'rb')),
                       pickle.load(open(OUTPUT_DIR + 'validation_domains_%s.list' % DATASET, 'rb')),
                       pickle.load(open(OUTPUT_DIR + 'test_domains_%s.list' % DATASET, 'rb'))
                       ):
    for domains in origin_domains:
        for domain in domains:
            all_len.append(len(domain['raw_domain']))

print(numpy.percentile(all_len, 0))
print(numpy.percentile(all_len, 50))
print(numpy.percentile(all_len, 60))
print(numpy.percentile(all_len, 70))
print(numpy.percentile(all_len, 80))
print(numpy.percentile(all_len, 90))
print(numpy.percentile(all_len, 100))