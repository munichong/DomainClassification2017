'''
Created on Oct 26, 2017

@author: munichong
'''
import csv
import json
from tldextract import extract
from wordsegment import load, segment
from nltk import word_tokenize


DMOZ_PATH = '../DMOZ/content.csv'
TRANS_DMOZ_PATH = '../DMOZ/transformed_content.json'

load()
output_table = {}


n = 0
with open(DMOZ_PATH, encoding='utf-8') as infile:
    csv_reader = csv.reader(infile)
    for line in csv_reader:
        raw_domain = line[0]

        print(n)
        n += 1
        
        if raw_domain in output_table:
            continue

        category_path = line[1].split('/')
        desc = ' '.join(line[2:]).replace(',', ' ')

        tld = extract(raw_domain)
        suffix = tld.suffix
        domain = '.'.join([tld.subdomain, tld.domain])
        
        output_table[raw_domain] = {
                                     'categories': category_path,
                                     'raw_domain': raw_domain,
                                     'domain': domain,
                                     'suffix': suffix,
                                     'segmented_suffix': suffix.split('.'),
                                     'segmented_domain': segment(domain),  # segment function is slow. Don't use for desc
                                     'tokenized_desc': word_tokenize(desc)
                                     }
#         print(output_table[raw_domain])
        
json.dump(output_table, open(TRANS_DMOZ_PATH, 'w'))

