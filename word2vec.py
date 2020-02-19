#import packages

import bs4
import urllib.request 
import re  
import nltk
from gensim.models import Word2Vec
import pandas as pd

res = urllib.request.urlopen('https://en.wikipedia.org/wiki/United_States')  
page = res.read()

text = bs4.BeautifulSoup(page,'lxml')

paragraphs = text.find_all('p')
content = ""

for p in paragraphs:  
    content = content + p.text
    
# Cleaing the text
processed_content = content.lower()  
processed_content = re.sub('[^a-zA-Z]', ' ', processed_content )  
processed_content = re.sub(r'\s+', ' ', processed_content)

# Preparing the dataset
all_sentences = nltk.sent_tokenize(processed_content)
all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words
from nltk.corpus import stopwords  
for i in range(len(all_words)):  
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]


word2vec = Word2Vec(all_words, min_count=2)  

word =  []
similarity = []
target = 'united'
if target in word2vec.wv.vocab:
    sim_words = word2vec.wv.most_similar(target)
    for i in sim_words:
        word.append(i[0])
        similarity.append(round(i[1],4)*100)
else:
    print('Word **', target, '** is not present')

pd.options.display.max_colwidth = 1000
df = pd.DataFrame(list(zip(word, similarity)), columns =['Word', 'Similarity(%)']) 
df
