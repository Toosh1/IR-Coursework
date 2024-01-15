import nltk
from nltk import bigrams
from nltk.probability import FreqDist
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
print(stops)