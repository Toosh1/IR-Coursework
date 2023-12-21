#Imports
import json
from math import log10 as log
from math import sqrt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
from nltk import PorterStemmer
from time import time
from spellchecker import SpellChecker
import spacy
from spacy import displacy


#Initialising
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()
stops = set(stopwords.words('english'))
vector_matrix = np.load("vector_matrix.npy")
spell = SpellChecker()

def eucLength(matrix):
    return sqrt(sum([x**2 for x in matrix])) #This calculates our ueclidean distance by squaring all the tf, ...
                                                  #... summing them, and then square rooting the sum.
    
def sim(doc1, doc2):    
    numerator = np.dot(doc1, doc2)  #np.dot() calculates the dot product between the vectors.
    doc1len = eucLength(doc1)
    doc2len = eucLength(doc2)

    denominator = doc1len * doc2len

    similarity = numerator / denominator
    return similarity

#File openings
with open("vocab.txt","r") as f:
    vocab = json.load(f)
with open("docIDs.txt","r") as f:
    docIDs = json.load(f)
with open("postings.txt","r") as f:
    postings = json.load(f)
with open("termCounter.txt","r") as f:
    numberOfTerms = json.load(f)

#Main Loop
while True:
    query_input = input("Search ('N' to cancel)::: ")
    t1 = time()
    # Check if cancel
    if query_input.lower() == "n":
        break


    query_tokens = word_tokenize(query_input)
    processed = nlp(query_input)
    for i in processed.ents:
        print(i.text)
        print(i.label_)
    
    

   #spellcheck
    misspelled = spell.unknown(query_tokens)
    for word in misspelled:
        corrected_word = spell.correction(word)
        query_tokens[query_tokens.index(word)] = corrected_word

    first = True
    scores_array = np.array([])
    query_tokens = [stemmer.stem(term) for term in query_tokens if term not in stops]
    #get number of vocabs
    postings_length = len(postings)
    query_vector = np.zeros((postings_length,1))
    for query_term in query_tokens:
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
        #get num of documents
        N = len(numberOfTerms)
        # number of docs containing term
        dft = len(postings_value)
        tfidf = (1 + log(N/dft)) * (1/len(query_tokens))
        query_vector[int(vocab_id)] = tfidf
    
    num_columns = vector_matrix.shape[1]
    results = []
    # Loop through columns
    for col_index in range(num_columns):
        column = vector_matrix[:, col_index]
        results.append(sim(column,query_vector)[0])
    

    #Arg sort document names with scores to get top ten names
    doc_ids_array = np.array(list(docIDs.values()))
    scores_array = np.array(results)
    sorted_scores = np.sort(results)[::-1]
    sorted_scores_arg = np.argsort(results)[::-1]
    sorted_web_searches = doc_ids_array[sorted_scores_arg[:10]] 

    for index,i in enumerate(sorted_web_searches):
        if (sorted_scores[index] > 0):
            print(i)
    t2 = time()
    print(str(t2-t1) + " s")




            
   