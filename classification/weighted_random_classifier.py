import pickle
from collections import defaultdict

OUTPUT_DIR = '../Output/'
DATASET = 'content'  # 'content' or '2340768'

domains_train = pickle.load(open(OUTPUT_DIR + 'training_domains_%s.list' % DATASET, 'rb'))
category_train_count = defaultdict(int)

for cat_domains in domains_train:
    top_level_cat = cat_domains[0]['categories'][1]
    category_train_count[top_level_cat] += len(cat_domains)

print(category_train_count)
total_train_domains = sum(category_train_count.values())
category_train_pct = {cat: count/total_train_domains for cat, count in category_train_count.items()}
print(category_train_pct)



domains_test = pickle.load(open(OUTPUT_DIR + 'test_domains_%s.list' % DATASET, 'rb'))
category_test_count = defaultdict(int)

for cat_domains in domains_test:
    top_level_cat = cat_domains[0]['categories'][1]
    category_test_count[top_level_cat] += len(cat_domains)
print(category_test_count)

accuracy = 0
for cat, count in category_test_count.items():
    accuracy += category_train_pct[cat] * count
print(accuracy / sum(category_test_count.values()))
