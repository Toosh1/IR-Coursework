#---------------Imports----------------#

from bs4 import BeautifulSoup

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk import PorterStemmer , bigrams

import os
import json
import regex as re
from time import time as t
from time import sleep

from math import log
import numpy as np

#----------------Initialising---------------#
t0 = t()
stops = set(stopwords.words('english'))
weights = {
    'title' : 5,
    'body' : 1,
    'h1' : 3,
    'h2' : 1.5,
    'h3' : 1.25,
}
stemmer = PorterStemmer()
lemmiter = WordNetLemmatizer()

#3 dictionaries to then save
vocab = {}
docIDs = {}
postings = {}
#postings include [docid,frequency,weightedFrequency,tf-idf value]

#---------------Functions------------------#


def openFiles():
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
    return content

def tokenizeFiles(content):
    tokens = []
    for num,i in enumerate(content):
        print("Tokenzing file " + str(num))
        page_tokens = complicatedTokenizor(i)
        tokens.append([(stemmer.stem(word[0].lower()), word[1], word[2]) for word in page_tokens])
        #tokens.append([(lemmiter.lemmatize(word[0].lower()), word[1], word[2]) for word in page_tokens])
    return tokens

def create_bigrams(tokens):
    bigrams_array = []
    bi_grams = list(bigrams(tokens))
    bi_gram_freq = FreqDist(bi_grams)
    bi_gram_dict = dict(bi_gram_freq)
    for key in bi_gram_dict:
        if bi_gram_dict[key] > 1:
            bigram_string = key[0] + " " + key[1]
            bigrams_array.append(bigram_string)
    return tokens + bigrams_array


def complicatedTokenizor(text):
    soup = BeautifulSoup(text, 'html.parser')
    content = soup.find(id="content")
    title = content.span
    zones = [title,content]
    frequencytuple = []
    for index,zone in enumerate(zones):
        cleanedTokens = []
        weightsArray = list(weights.values())
        for p in zone:
            cleanedParagraphs = p.get_text()
            cleaned_string = re.sub(r'[^\w\s-]', '', str(cleanedParagraphs))
            cleaned_string = re.sub(r'-', ' ', cleaned_string)
            tokens = word_tokenize(cleaned_string)
            cleanedTokens += [t.lower() for t in tokens if t not in stops]
        
        cleanedTokens = create_bigrams(cleanedTokens)
        for token in cleanedTokens:
            done = False
            counter = 0
            #bigrams will have a weight of 4 to promote searching for them
            if " " in token:
                isBigram = 4
            else:
                isBigram = 1
            for frequencyToken,_1,_2 in frequencytuple:
                if token == frequencyToken:

                    tempUpdate = list(frequencytuple[counter])
                    tempUpdate[1] += 1
                    tempUpdate[2] += weightsArray[index] * isBigram
                    tempUpdate = tuple(tempUpdate)
                    frequencytuple[counter] = tempUpdate
                    done = True
                    break
                counter += 1
            if not done:
                frequencytuple.append((token,1,weightsArray[index] * isBigram))
            
    return frequencytuple


def inverseIndex(tokens):
    counter = 0
    docIDcounter = 0
   

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

def createVectorMatrix():
    vector_matrix = np.zeros((len(postings),int(len(numberOfTerms))))

    for index,terms in enumerate(postings.values()):
        print("Adding tf-idf matrix " + str(index))
        N = len(numberOfTerms)
        dft = len(terms)
        for doc in terms:
            tfdt = doc[2] / numberOfTerms[doc[0]]
            idf_value = tfdt * (1 + log(N / dft))
            vector_matrix[index,doc[0]] = idf_value
    np.save("vector_matrix",vector_matrix)

def saveFiles():
    global vocab,docIDs,postings,numberOfTerms
    with open("vocab.txt", "w") as f:
        json.dump(vocab, f)

    with open("docIDs.txt", "w") as f:
        json.dump(docIDs, f)

    with open("postings.txt", "w") as f:
        json.dump(postings, f)

    with open("termCounter.txt", "w") as f:
        json.dump(numberOfTerms, f)



#------- Main -------#
content = openFiles()
tokens = tokenizeFiles(content)
numberOfTerms = [0] * len(docIDs)
inverseIndex(tokens)
createVectorMatrix()
saveFiles()


t1 = t()
print(str(t1 - t0) + "s")