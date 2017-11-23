'''
Created on Oct 26, 2017

@author: munichong
'''
import csv
import json
from tldextract import extract
from wordsegment import load, segment


DMOZ_PATH = '../DMOZ/parsed-new_2340768.csv'
TRANS_DMOZ_PATH = '../DMOZ/transformed_parsed-new_2340768.json'

load()
output_table = {}


n = 0
with open(DMOZ_PATH) as infile:
    csv_reader = csv.reader(infile)
    for line in csv_reader:
        raw_domain = line[0]
        category_path = line[1].split('/')
        
        if n % 100000 == 0:
            print(n)
        n += 1
        
        if raw_domain in output_table:
            continue

        tld = extract(raw_domain)
        suffix = tld.suffix
        domain = '.'.join([tld.subdomain, tld.domain])
        segmented_domain = segment(domain)
        
        output_table[raw_domain] = {
                                     'categories': category_path,
                                     'raw_domain': raw_domain,
                                     'domain': domain,
                                     'suffix': suffix,
                                     'segmented_suffix': suffix.split('.'),
                                     'segmented_domain': segmented_domain,
                                     }
#         print(output_table[raw_domain])
        
json.dump(output_table, open(TRANS_DMOZ_PATH, 'w'))

