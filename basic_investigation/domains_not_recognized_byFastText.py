import pickle
from gensim.models.wrappers import FastText

OUTPUT_DIR = '../Output/'

# Creating the model
print("Loading the FastText Model")
# en_model = {}
en_model = FastText.load_fasttext_format('../FastText/wiki.en/wiki.en')

domains_train = pickle.load(open(OUTPUT_DIR + 'training_domains.list', 'rb'))

n_total = 0
n_0segment = 0
for domain in domains_train:
    embeds = [en_model[w].tolist() for w in domain['segmented_domain'] if w in en_model]
    if not embeds:
        n_0segment += 1
    n_total += 1

print('n_0segment =', n_0segment)
print('n_total =', n_total)
print(n_0segment / n_total)
