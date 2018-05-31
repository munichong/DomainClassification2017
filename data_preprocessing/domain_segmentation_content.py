'''
Created on Oct 26, 2017

@author: munichong
'''
import csv, pickle, re
import json
from tldextract import extract
from wordsegment import load, segment
from nltk import word_tokenize
from urllib.parse import urlparse


DMOZ_PATH = '../DMOZ/content.csv'
TRANS_DMOZ_PATH = '../DMOZ/transformed_content.pkl'

load()
output_table = {}


n = 0
with open(DMOZ_PATH, encoding='utf-8') as infile:
    csv_reader = csv.reader(infile)
    for line in csv_reader:
        raw_domain = line[0]

        if n % 100000 == 0:
            print(n)
        n += 1

        # label duplicate or ambiguous domains
        if raw_domain in output_table:
            output_table[raw_domain] = {}
            continue

        # raw_domain = 'http://www.the401kman.com/'
        tld = extract(raw_domain)
        subdomain, domain, suffix = tld.subdomain, tld.domain, tld.suffix
        # domain = '.'.join([tld.subdomain, tld.domain])
        # print(tld.subdomain, tld.domain)

        # Remove domains with subdomains such as "http://bakery.cdkitchen.com/"
        # Do not remove domains such as "http://mikegrost.com/"
        if (subdomain and subdomain != 'www') and len(re.findall('(?=\.)', raw_domain)) > 1:
            # print(raw_domain)
            output_table[raw_domain] = {}
            continue

        url_seg = urlparse(raw_domain)
        if url_seg.path != '/' or url_seg.query or url_seg.fragment:
            output_table[raw_domain] = {}
            continue

        category_path = line[1].split('/')
        desc = ' '.join(line[2:]).replace(',', ' ')

        # http://www.e-scanshop.com/ ---> ['e', 'scan', 'shop']
        #                             not ['es', 'can', 'shop']
        if '-' in domain:
            for s in domain.split('-'):
                segmented_domain.extend(segment(s))
        else:
            segmented_domain = segment(domain) # segment function is slow. Don't use for desc


        output_table[raw_domain] = {
                                     'categories': category_path,
                                     'raw_domain': raw_domain,
                                     'domain': domain,
                                     'suffix': suffix,
                                     'segmented_suffix': suffix.split('.'),
                                     'segmented_domain': segmented_domain,
                                     'tokenized_desc': word_tokenize(desc)
                                     }


num_total_domains = len(output_table)
filtered_domains = {domain: output_table[domain] for domain in output_table if output_table[domain]}
num_filtered_domains = len(filtered_domains)
print("%d domains will be pickled." % num_filtered_domains)
print('%d (%.4f) domains are duplicate, non-homepage, and/or ambiguous' % (num_total_domains - num_filtered_domains,
                                                            1 - (num_filtered_domains / num_total_domains)))

# pickle.dump(filtered_domains, open(TRANS_DMOZ_PATH, 'wb'))
