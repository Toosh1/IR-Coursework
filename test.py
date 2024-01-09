import nltk
from nltk import bigrams
from nltk.probability import FreqDist

# Sample long text
long_text = "Your long text goes here star wars. Replace this with your actual text. star wars"

# Tokenize the text into words
words = nltk.word_tokenize(long_text)

# Create bigrams from the list of words
bi_grams = list(bigrams(words))

# Calculate the frequency distribution of bigrams
bi_gram_freq = FreqDist(bi_grams)

# Convert the frequency distribution to a dictionary
bi_gram_dict = dict(bi_gram_freq)

# Print the dictionary of bigram frequencies
print(bi_gram_dict)
