from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
import os
import json
import regex as re
from time import time as t
from math import log
stops = set(stopwords.words('english'))
t0 = t()
weights = {
    'title' : 5,
    'body' : 1,
    'h1' : 3,
    'h2' : 1.5,
    'h3' : 1.25,
    
}


def simpleTokenizor(text):
    soup = BeautifulSoup(text, 'html.parser')
    title = soup.title()
    paragraphs = soup.find_all('body')
    h1Tags = soup.find_all("h1")
    h2Tags = soup.find_all("h2")
    h3Tags = soup.find_all("h3")
    
    zones = [title,paragraphs,h1Tags,h2Tags,h3Tags]

    frequencytuple = []
    for index,zone in enumerate(zones):
        cleanedTokens = []
        
        weightsArray = list(weights.values())
        for p in zone:
            cleanedParagraphs = p.get_text()

            reS1 = re.sub("[^\w\s]", "", str(cleanedParagraphs))
            tokens = word_tokenize(reS1)
            cleanedTokens += [t.lower() for t in tokens if t not in stops]
            if index == 2:
                print(cleanedTokens)

            for token in cleanedTokens:
                done = False
                counter = 0
                for frequencyToken, frequency,weightedFrequency in frequencytuple:
                    if token == frequencyToken:

                        tempUpdate = list(frequencytuple[counter])
                        tempUpdate[1] += 1
                        tempUpdate[2] += weightsArray[index]
                        tempUpdate = tuple(tempUpdate)
                        frequencytuple[counter] = tempUpdate
                        done = True
                        break
                    counter += 1
                if not done:
                    frequencytuple.append((token,1,weightsArray[index]))
    return frequencytuple



stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

vocab = {}
docIDs = {}
postings = {}
# postings include [docid,frequency,weightedFrequency,tf-idf value]

folder_name = "videogames\\"
file_names = [f for f in os.listdir("videogames") if os.path.isfile(os.path.join("videogames", f))]
content = []
tokens = []

counter = 0
for file_name in file_names:
    print("Reading file "+ str(counter))
    docIDs[counter] = file_name
    with open(folder_name + file_name, 'r') as file_handle:
        content.append(file_handle.read())
    counter += 1

for num,i in enumerate(content):
    print("Tokenzing file " + str(num))
    page_tokens = simpleTokenizor(i)
    tokens.append([(stemmer.stem(word[0].lower()), word[1], word[2]) for word in page_tokens])
counter = 0
docIDcounter = 0
numberOfTerms = [0] * len(docIDs)

for docTokens in tokens:
    for token, frequency,weightedFrequency in docTokens:
        if token in vocab.values():
            # update postings with new docid and frequency
            postingsKey = list(vocab.values()).index(token)
            docIDsPosted = [i for i, f, wf in postings.get(postingsKey, [])]
            if docIDcounter not in docIDsPosted:
                postingsValue = postings.get(postingsKey, [])
                postingsValue.append([docIDcounter, frequency,weightedFrequency])
                postings[postingsKey] = postingsValue
                numberOfTerms[docIDcounter] += 1
        else:
            # add value to vocab as not yet present
            numberOfTerms[docIDcounter] += 1
            vocab[counter] = token
            # add new index to postings
            postings[counter] = [[docIDcounter, frequency,weightedFrequency]]
            counter += 1
    print("Inverse Indexing " + str(docIDcounter))
    docIDcounter += 1


for index,terms in enumerate(postings.values()):
    print("Adding tf-idf matrix " + str(index))
    N = len(numberOfTerms)
    dft = len(terms)
    for doc in terms:
        tfdt = doc[2] / numberOfTerms[doc[0]]
        doc.append(tfdt * (1 + log(N / dft)))





with open("vocab.txt", "w") as f:
    json.dump(vocab, f)

with open("docIDs.txt", "w") as f:
    json.dump(docIDs, f)

with open("postings.txt", "w") as f:
    json.dump(postings, f)
with open("termCounter.txt", "w") as f:
    json.dump(numberOfTerms, f)
t1 = t()

print(str(t1 - t0) + "s")