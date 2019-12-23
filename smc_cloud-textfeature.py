import sys
import re
import pandas as pd
import nltk; nltk.download('stopwords')
import gensim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.utils import simple_preprocess

### Read the datafile
filename = sys.argv[1]
df = pd.read_csv(open(filename),usecols=['text'])

### Text pre-proprocessing steps
### Convert to list
texts = df.text.values.tolist()

### Remove new line characters
texts = [re.sub('\s+', ' ', x) for x in texts]

### Remove Emails and Mentions
texts = [re.sub('\S*@\S*\s?', '', x) for x in texts]
texts = [re.sub('@\S*\s?', '', x) for x in texts]
# @USER

### Remove distracting single quotes
texts = [re.sub("\'", "", x) for x in texts]

### Remove URLs
texts = [re.sub(r"http\S+", "", x) for x in texts]

### Extract binary count feature
vec = CountVectorizer(binary=True)
vec.fit(texts)

feature_binary = pd.DataFrame(vec.transform(texts).toarray(), columns=sorted(vec.vocabulary_.keys()))
feature_binary

### Extract count feature
vec = CountVectorizer(binary=False) # we cound ignore binary=False argument since it is default
vec.fit(texts)

feature_count = pd.DataFrame(vec.transform(texts).toarray(), columns=sorted(vec.vocabulary_.keys()))
feature_count

### Extract tfidf feature
vec = TfidfVectorizer()
vec.fit(texts)

feature_tfidf = pd.DataFrame(vec.transform(texts).toarray(), columns=sorted(vec.vocabulary_.keys()))
feature_tfidf

### Save text feature dataframe into CSV
feature_binary.to_csv('./textfeature_binary.csv')
feature_count.to_csv('./textfeature_count.csv')
feature_tfidf.to_csv('./textfeature_tfidf.csv')
