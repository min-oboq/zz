import numpy as np
%matplotlib notebook
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import keyedvectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('data\glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = keyedvectors.load_word2vec_format(word2vec_glove_file)
model.most_similar('bill')

model.most_similar('cherry')

model.most_similar(negative=['cherry'])

result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))

def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]
analogy('australia', 'beer', 'france')

analogy('tall', 'tallest', 'long')

print(model.doesnt_match("breakfast cereal dinner lunch".split()))

