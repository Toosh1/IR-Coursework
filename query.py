import json
from math import log10 as log
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from nltk import PorterStemmer
stemmer = PorterStemmer()
stops = set(stopwords.words('english'))
from time import time
from bs4 import BeautifulSoup
def return_test_text(doc,tokens):
    folder_name = "videogames\\"
    with open(folder_name + doc, 'r') as file_handle:
        text = file_handle.read()
        soup = BeautifulSoup(text, 'html.parser')
        paragraphs = soup.find(id = 'content')
        title = paragraphs.span.getText()
        print(title)
        #print("C:/Users/Mateusz/Documents/GitHub/IR-Coursework/videogames/" +doc)


with open("vocab.txt","r") as f:
    vocab = json.load(f)
with open("docIDs.txt","r") as f:
    docIDs = json.load(f)
with open("postings.txt","r") as f:
    postings = json.load(f)
with open("termCounter.txt","r") as f:
    numberOfTerms = json.load(f)

games = [
    "Action Adventure Games",
    "RPG Games for PlayStation",
    "The Guy Game",
    "Spider Man",
    "star wars battlefront",
    "Iron Man",
    "James Earl Cash",
    "Crazy Taxi",
    "James Bond",
    "The Lord of the Rings The Two Towers PS2",
    "n"
    ] 
for i in games:
    query_input = input(":::")
    t1 = time()
    # Check if cancel
    if query_input.lower() == "n":
        break

    query_array = query_input.split(" ")
    first = True
    scores_array = np.array([])
    for query_term in query_array:
        if query_term not in stops:
            query_term = stemmer.stem(query_term)
            # get vocab id from dict
            vocab_id = ""
            for key, value in vocab.items():
                if query_term.lower() == value:
                    vocab_id = key
                    break

            # exit if not found
            if vocab_id == "":
                print(f"Cannot find term '{query_term}'")
                continue  # Skip to the next term

            # get postings value (array with term frequency for each document)
            postings_value = postings.get(vocab_id)
            #get names of documents
            docNames = list(docIDs.values())
            # set doc scores beforehand
            N = len(numberOfTerms)
            doc_scores = [0] * N
            # Calculate scores for querySet1
            dft = len(postings_value)

            for doc_id, frequency, weightedFrequency in postings_value:
                tfdt = weightedFrequency / numberOfTerms[doc_id]
                doc_scores[doc_id] = tfdt * (1 + log(N / dft))
            if first:
                scores_array = np.array(doc_scores)
                first = False
            else:
                scores_array = np.multiply(scores_array,doc_scores)
                
    doc_ids_array = np.array(list(docIDs.values()))
    scores_array = np.array(scores_array)
    sorted_scores = np.sort(scores_array)[::-1]
    sorted_scores_arg = np.argsort(scores_array)[::-1]
    sorted_web_searches = doc_ids_array[sorted_scores_arg[:]]
    t2 = time()
    document_count = 0
    for index,i in enumerate(sorted_web_searches):
        if (sorted_scores[index] > 0):
            return_test_text(i,1)
            document_count += 1



 # # Calculate scores for the entire document collection
    # doc_scores_total = [0] * N
    # for posting in postings.values():
    #     if posting is not None:
    #         dft_total = len(posting)
    #         for doc_id, frequency, weightedFrequency in posting:
    #             tfdt_total = weightedFrequency / numberOfTerms[doc_id]
    #             doc_scores_total[doc_id] += tfdt_total * (1 + log(N / dft_total))