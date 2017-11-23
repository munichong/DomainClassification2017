'''
Created on Oct 26, 2017

@author: munichong
'''
import json, pickle, os
from pprint import pprint
from random import shuffle
from collections import Counter, defaultdict

DATASET = 'content' # 'content' or '2340768'

TRANS_DMOZ_PATH = '../DMOZ/transformed_%s.json' % DATASET

OUTPUT_DIR = '../Output/'
TRAIN_PCT = 0.8


json_data = json.load(open(TRANS_DMOZ_PATH))


"""Group the domains that belong to the same categories. """
category2index = defaultdict(int)
suffix2index = defaultdict(int)
suffix_dist = Counter()
category_domains = [[]]
domain_segments_len_dist = Counter()
for full_domain in json_data:

    ''' processing target category '''
    second_category = json_data[full_domain]['categories'][1]
    if second_category == 'World' or second_category == 'Regional':
        continue

    if second_category not in category2index:
        category2index[second_category] = len(category2index) # class index starts from 0
        category_domains.append([])
    target = category2index[second_category]
    json_data[full_domain]['target'] = target


    ''' processing suffix '''
    seg_suffix = json_data[full_domain]['segmented_suffix']

    suffix_indices = []
    for suffix in seg_suffix:
        if suffix not in suffix2index:
            suffix2index[suffix] = len(suffix) + 1 # suffix index starts from 1
        suffix_indices.append(suffix2index[suffix])
        suffix_dist[suffix] += 1

    json_data[full_domain]['suffix_indices'] = suffix_indices


    # record the length of the segmented domain
    domain_segments_len_dist[len(json_data[full_domain]['segmented_domain'])] += 1



    category_domains[target].append(json_data[full_domain])



print("The Distribution of Categories Sizes:")
total = 0
for category in category2index:
    category_size = len(category_domains[category2index[category]])
    print(category, category2index[category], ":", category_size)
    total += category_size
print("There are", total, "domains in total.")
print()


print("The Distribution of domains segments lengths:")
for length, count in domain_segments_len_dist.items():
    print(length, count)
print()


pickle.dump(category2index, open(os.path.join(OUTPUT_DIR + 'category2index_%s.dict' % DATASET), 'wb'))
pickle.dump(suffix2index, open(os.path.join(OUTPUT_DIR + 'suffix2index_%s.dict' % DATASET), 'wb'))


""" Split training, validation, and test datasets """
training_domains = []
validation_domains = []
test_domains = []
for domains in category_domains:
    shuffle(domains)
    train_end_index = int(len(domains) * TRAIN_PCT)
    training_domains.extend(domains[ : train_end_index])
    validation_end_index = int(len(domains) * (TRAIN_PCT + (1 - TRAIN_PCT) / 2))
    validation_domains.extend(domains[train_end_index : validation_end_index] )
    test_domains.extend(domains[validation_end_index: ])
print("Training Size:", len(training_domains), "Validation Size:", len(validation_domains), "Test Size:", len(test_domains))
pickle.dump(training_domains, open(os.path.join(OUTPUT_DIR + 'training_domains_%s.list' % DATASET), 'wb'))
pickle.dump(validation_domains, open(os.path.join(OUTPUT_DIR + 'validation_domains_%s.list' % DATASET), 'wb'))
pickle.dump(test_domains, open(os.path.join(OUTPUT_DIR + 'test_domains_%s.list' % DATASET), 'wb'))


params = {'num_targets': len(category2index), 'num_suffix': len(suffix2index), 'max_domain_segments_len': max(domain_segments_len_dist.keys())}
print(params)
json.dump(params, open(os.path.join(OUTPUT_DIR, 'params_%s.json' % DATASET), 'w'))

pprint(suffix_dist.most_common())
print(len(suffix_dist.most_common()))