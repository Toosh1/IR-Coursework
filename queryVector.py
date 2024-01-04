#Imports
import json
from math import log10 as log
from math import sqrt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, pos_tag
import numpy as np
from nltk import PorterStemmer
from time import time
from spellchecker import SpellChecker
import spacy
from spacy import displacy
from colorama import Fore, Back, Style
from bs4 import BeautifulSoup
import textdistance

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

def returnText(doc,tokens):
    folder_name = "videogames\\"
    with open(folder_name + doc, 'r') as file_handle:
        text = file_handle.read()
        soup = BeautifulSoup(text, 'html.parser')
        paragraphs = soup.find(id = 'content')
        text = paragraphs.get_text()
        title = paragraphs.span.getText()
        text = text.split("\n")
        first_paragrpah = text[12]
        print(Style.BRIGHT + title)
        print(Style.RESET_ALL + Fore.BLUE + (folder_name+doc))
        print(Fore.RESET + Style.DIM + first_paragrpah[:250] + "...")
        print(Style.RESET_ALL)

def findClosestWord(word):
    max_distance = 0
    closest_word = ''
    for i in vocab.values():
        distance = textdistance.jaro_winkler(word,i)
        if distance>max_distance:
            max_distance = distance
            closest_word = i
    return closest_word

def openFiles():
    global vocab, docIDs, postings, numberOfTerms
    #File openings
    with open("vocab.txt","r") as f:
        vocab = json.load(f)
    with open("docIDs.txt","r") as f:
        docIDs = json.load(f)
    with open("postings.txt","r") as f:
        postings = json.load(f)
    with open("termCounter.txt","r") as f:
        numberOfTerms = json.load(f)

def print_results(results,query_tokens):
    global t1
    #Arg sort document names with scores to get top ten names
    doc_ids_array = np.array(list(docIDs.values()))
    scores_array = np.array(results)
    sorted_scores = np.sort(results)[::-1]
    sorted_scores_arg = np.argsort(results)[::-1]
    sorted_web_searches = doc_ids_array[sorted_scores_arg] 

    results = 0
    for index,i in enumerate(sorted_web_searches):
        if (sorted_scores[index] > 0):
            results += 1
    t2 = time()
    print(str(results) +  " results in " + str(round((t2-t1),6)) + " s")


    for index,i in enumerate(sorted_web_searches[:10]):
        if (sorted_scores[index] > 0):
            returnText(i,query_tokens)

def rank_function(query_tokens):
    first = True
    scores_array = np.array([])
    query_tokens = [(stemmer.stem(term),weight) for term,weight in query_tokens if term not in stops]
    #get number of vocabs
    postings_length = len(postings)
    query_vector = np.zeros((postings_length,1))
    for query_pair in query_tokens:
        query_term, weight = query_pair
        # get vocab id from dict
        vocab_id = ""
        for key, value in vocab.items():
            if query_term.lower() == value:
                vocab_id = key
                break

        # exit if not found
        if vocab_id == "":
            #if vocab not found print you cant find it find closest one and append new vocab to query
            closest_vocab = findClosestWord(query_term)
            print(f"Did you mean '{closest_vocab}'?")
            query_tokens.append(stemmer.stem(closest_vocab))
            continue  # Skip to the next term
        
        # get postings value (array with term frequency for each document)
        postings_value = postings.get(vocab_id)
        #get num of documents
        N = len(numberOfTerms)
        # number of docs containing term
        dft = len(postings_value)
        tfidf = (1 + log(N/dft)) * (1/len(query_tokens))
        query_vector[int(vocab_id)] = tfidf * weight #weight is for thesauras (expanded query should be worth less)

    num_columns = vector_matrix.shape[1]
    results = []
    # Loop through columns
    for col_index in range(num_columns):
        column = vector_matrix[:, col_index]
        results.append(sim(column,query_vector)[0])
    return results

def pos_tagging(query):
    #tag certain part of the speech to add weights to important aspects
    processed = nlp(query)
    for i in processed.ents:
        print(i.text)
        print(i.label_)

def get_synonyms(word):
    #use sysnet to get synonyms of a word
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def rank_synonyms(original_word, synonyms):
    ranked_synonyms = []
    #take the original word and synonym and compare similairity to then weight how useful it is for a search
    for synonym in synonyms:
        similarity = wordnet.synsets(original_word)[0].wup_similarity(wordnet.synsets(synonym)[0])
        if similarity is not None:
            if similarity != 1 and similarity >= 0.5:
                #if = 1 then same word and below 0.5 word is too dissimilair 
                ranked_synonyms.append((synonym,similarity/5)) # divide by 5 as to reduce weight to actual query term (1)
    ranked_synonyms = sorted(ranked_synonyms, key=lambda x: x[1], reverse=True)
    return ranked_synonyms

def produce_weighted_synonym_list(query):
    tokens = word_tokenize(query)
    pos_tags = pos_tag(tokens)
    pos_tags = [token for token in pos_tags if token[0] not in stops]
    noun_tokens = [token[0] for token in pos_tags if token[1] == 'NN']
    #produce synonums for all nouns in a query
    all_synonyms = []
    for noun in noun_tokens:
        synonyms = get_synonyms(noun)
        ranked_synonyms = rank_synonyms(noun,synonyms)
        for i in ranked_synonyms[:3]:
            all_synonyms.append(i)
        #append only top 3 as to not over search too many queries
    return all_synonyms

def add_standard_weights(query_tokens):
    #thesauras tokens will have a different weigh however query tokens all should
    #have a weight of 1 as they are what the user requested
    return [(x,1) for x in query_tokens]


def main_loop():
    global t1
    while True:
        query_input = input("Search ('N' to cancel)::: ")
        t1 = time()
        # Check if cancel
        if query_input.lower() == "n":
            break

        query_tokens = word_tokenize(query_input)
        query_tokens = add_standard_weights(query_tokens)
        synonym_tokens = produce_weighted_synonym_list(query_input)
        for i in synonym_tokens:
            query_tokens.append(i)
        results = rank_function(query_tokens)
        print_results(results,query_tokens)
  
                
#Main Loop
openFiles()
main_loop()



                
