import pandas as pd
class2 = pd.read_csv('data\class2.csv')

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

train_x = label_encoder.fit_transform(class2['class2'])
train_x

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is last chance.',
    'and if you do not have this chance',
    'you will never get any chance',
    'will you do get this one?',
    'please, get this chance',
]
vect = CountVectorizer()
vect.fit(corpus)
vect.vocabulary_

vect.transform(['you will never get any chance.']).toarray()

vect = CountVectorizer(stop_words=["and", "is", "please", "this"]).fit(corpus)
vect.vocabulary_

from sklearn.feature_extraction.text import TfidfVectorizer
doc = ['I like machine learning', 'I love deep learning', 'I run everyday']
tfidf_vectorizer = TfidfVectorizer(min_df=1)
tfidf_matrix = tfidf_vectorizer.fit_transform(doc)
doc_distance = (tfidf_matrix * tfidf_matrix.T)
print('유사도를 위한', str(doc_distance.get_shape()[0]), 'x', str(doc_distance.get_shape()[1]), '행렬을 만들었습니다.')
print(doc_distance.toarray())

from nltk.tokenize import sent_tokenize, word_tokenize
import Warnings
warnings.filterwarnings(action='ignore')
import gensim
from gensim.models import Word2Vec

sample = open("data\peter.txt", "r", encoding='UTF8')
s = sample.read()

f = s.replace("\n", " ")
data = []

for i in sent_tokenize(f):
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)
data

modl1 = gensim.models.Word2Vec(data, min_count=1, vect_size=100, window=5, sg=0)
print("Cosine similarity between 'peter' " + "'wendy' - CBOW : ", model1.wv.similarity('peter', 'wendy'))

print("Cosine similarity between 'peter' " + "'hook' - CBOW : ", model1.wv.similarity('peter', 'hook'))

model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)
print("Cosine similarity between 'peter' " + "'wendy' - Skip Gram : ", model1.wv.similarity('peter', 'wendy'))

print("Cosine similarity between 'peter' " + "'hook' - Skip Gram : ", model1.wv.similarity('peter', 'hook'))

from gensim.test.utils import common_texts
from gensim.models import FastText

model = FastText('data\peter.txt', vector_size=4, window=3, min_count=1, epochs=10)
sim_score = model.wv.similarity('peter', 'wendy')
print(sim_score)

sim_score = model.wv.similarity('peter', 'hook')
print(sim_score)

find_similar_to = '노력'

for similar_word in model_kr.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(similar_word[0], similar_word[1]))
    
similarities = model_kr.most_similar(positive=['동물', '육식동물'], negative=['사람'])
print(similarities)