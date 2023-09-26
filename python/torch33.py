# 1 
from nltk import sent_tokenize
text_sample = '''
Share full article


About a half dozen writers and actors carrying signs, and some wearing hats, picket outside Paramount studios in Los Angeles.
Striking Hollywood writers and actors picketing outside Paramount Studios in Los Angeles last week.Credit...Mario Tama/Getty Images

Brooks BarnesJohn Koblin
By Brooks Barnes and John Koblin
Sept. 25, 2023, 1:09 a.m. ET
'''
tokenized_sentences = sent_tokenize(text_sample)
print(tokenized_sentences)

# 2
from nltk import word_tokenize
sentence = "The faster Harry got to the store, the faster Harry, the faster, would get home."
words = word_tokenize(sentence)
words

# 3 
from nltk.tokenize import WordPunctTokenizer
sentence = "it's nothing that you don't already know the answer to most people aren't aware of how their inner world works."
words = WordPunctTokenizer().tokenize(sentence)
words

# 4
import csv
from konlpy.tag import Okt
from gensim.models import word2vec

f = open('data/ratings_train.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
rdw = list(rdr)
f.close()

# 5
twitter = Okt()

result = []
for line in rdw:
    malist = twitter.pos(line[1], norm=True, stem=True)
    r = []
    for word in malist:
        if not word[1] in ['Josa', 'Eomi', 'Punctuation']:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    result.append(rl)
    print(rl)

# 6
with open("NaverMovie.nlp", 'w', encoding='utf-8') as fp:
    fp.write("\n".join(result))
    
# 7
mData = word2vec.LineSentence("NaverMovie.nlp")
mModel = word2vec.Word2Vec(mData, vector_size=200, window=10, hs=1, min_count=2, sg=1)
mModel.save("NaverMovie.model")


# 8 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

sample_text = "One of first things I noticed about the book was the unusual way author chose to narrate story. Instead of telling it from perspective of one or two characters, author chose to tell it from perspective of an entire town."
text_tokens = word_tokenize(sample_text)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
print(f'불용어 제거 미적용: {text_tokens}', '\n')
print(f'불용어 제거 적용: {tokens_without_sw}')

# 9
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

print(stemmer.stem('obsesse'), stemmer.stem('obsessed'))
print(stemmer.stem('standardizes'), stemmer.stem('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))

# 10 lancaster
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

print(stemmer.stem('obsesse'), stemmer.stem('obsessed'))
print(stemmer.stem('standardizes'), stemmer.stem('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))

# 11 wordnet
import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
print(stemmer.stem('obsesse'), stemmer.stem('obsessed'))
print(lemma.lemmatize('obsesse'), lemma.lemmatize('obsessed'))
print(stemmer.stem('standardizes'), stemmer.stem('standardization'))
print(lemma.lemmatize('standardizes'), lemma.lemmatize('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))
print(lemma.lemmatize('national'), lemma.lemmatize('nation'))
print(stemmer.stem('saying'), stemmer.stem('said'))
print(lemma.lemmatize('saying'), lemma.lemmatize('said'))
print(stemmer.stem('university'), stemmer.stem('universe'))
print(lemma.lemmatize('university'), lemma.lemmatize('universe'))

print(lemma.lemmatize('obsesse', 'v'), lemma.lemmatize('obsessed', 'a'))
print(lemma.lemmatize('standardizes', 'v'), lemma.lemmatize('standardization', 'n'))
print(lemma.lemmatize('national', 'a'), lemma.lemmatize('nation', 'n'))
print(lemma.lemmatize('saying', 'r'), lemma.lemmatize('said'), 'v')
print(lemma.lemmatize('university', 'r'), lemma.lemmatize('universe', 'n'))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('diabetes.cvs')
x = df[df.columns[:-1]]
y = df['Outcome']

x = x.values
y = torch.tensor(y.values)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

ms = MinMaxScaler()
ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ms.fit_transform(y_train)
y_test = ms.fit_transform(y_test)

class Customdataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
    
train_data = Customdataset(torch.FloatTensor(x_train),torch.FloatTensor(y_train))
train_data = Customdataset(torch.FloatTensor(x_test),torch.FloatTensor(y_test))

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=False)

class BinaryClassification(nn.Module):
    def _init(self):
        super(BinaryClassification, self).init__()
        self.layer_1 = nn.Linear(8, 64, bias=True)
        self.layer_2 = nn.Linear(64, 64, bias=True)
        self.layer_out = nn.Linear(64, 1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnormt = nn. BatchNorm1d(64)
        self.batchnorm2= nn.BatchNormld(64)
        
    def forward(self, inputs):
        x= self.relu(self.layer_1(inputs))
        x= self.batchnorm1(x)
        x= self.relu(self.layer2(x))
        x= self.batchorm2()
        x= self.dropout(x)
        x= self.layer_out(y)
        return x
    
epochs =1000+1
print_epoch = 100
LEARNING_RATE =1e-2

model = BinaryClassification()
model.to(device)
print(model)
BCE = nn.BCEWithlogitsloss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

for epoch in range(epochs):
    iteration_loss = 0.
    iteration_accuracy = 0.
    
    model.train()
    for i, data in enumerate(train_loader):
        x, y = data
        y_pred = model(x.float())
        loss = BCE(y_pred, y.reshape(-1,1).float())
        
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(epoch % print_epoch == 0):
            print('Train: epoch {0} - loss: {1:.5f}; acc: {2:.3f}'.format(epoch, iteration_loss/(i+1), iteration_accuracy/(i+1)))
        
        iteration_loss = 0.
        iteration_accuracy = 0.
        model.eval()
        for i, data in enumerate(test_loader):
            x, y = data
            y_pred = model(x.float())
            loss = BCE(y_pred, y.reshape(-1,1).float())
            iteration_loss += loss
            iteration_accuracy += accuracy(y_pred, y)
        if(epoch % print_epoch == 0):
            print('Train: epoch {0} - loss: {1:.5f}; acc: {2:.3f}'.format(epoch, iteration_loss/(i+1), iteration_accuracy/(i+1)))