import numpy as np
from pprint import pprint
import csv
from collections import defaultdict, Counter

root_path = '/Users/chong.wang/Dropbox/'

pred_table = defaultdict(Counter)
total_counter = Counter()
with open(root_path + 'all_predictions_frozen.csv') as infile:
    csv_reader = csv.reader(infile)
    next(csv_reader)
    next(csv_reader)
    for line in csv_reader:
        y_true, y_pred = line[2], line[3]
        pred_table[y_true][y_pred] += 1
        total_counter[y_true] += 1

        if y_true == 'Shopping' and y_pred == 'Business':
            print(line[:-1])
print(pred_table)
pprint(total_counter)


for y_true, Y_pred in pred_table.items():
    for y_pred in Y_pred:
        pred_table[y_true][y_pred] /= total_counter[y_true]
pprint(pred_table)