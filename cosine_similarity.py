from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.corenlp import CoreNLPParser
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math


# same readData from STS.py
def readData():

    first_sentence = []
    second_sentence = []
    score = []
    fileName = "./data/train-set.txt"
    file = open(fileName, encoding="utf8")
    text = file.readline()
    text = file.read()
    # loop to extract a set of two sentences
    for sentence in text.split('\n'):
        # creating two separate lists of the sentences
        # '.rstrip('.') only removes the last period in the sentence
        first_sentence.insert(len(first_sentence),
                              (sentence.split('\t')[1].lower()).rstrip('.'))
        second_sentence.insert(len(first_sentence),
                               (sentence.split('\t')[2].lower()).rstrip('.'))
        # inserting the score as a separate lists
        score.insert(len(first_sentence), (sentence.split('\t')[3]))

    # print(first_sentence)
    return first_sentence, second_sentence, score


def preprocess():

    first_sentence, second_sentence, score = readData()
    first_sentence_tokens = []
    second_sentence_tokens = []

    # tokenizing and tagging
    first_sentence_tags = []
    second_sentence_tags = []

    for sentence in first_sentence:
        tokens = nltk.word_tokenize(sentence)
        first_sentence_tokens.insert(len(first_sentence_tokens), tokens)
        first_sentence_tags.insert(
            len(first_sentence_tags), nltk.pos_tag(tokens))
        # print(first_sentence_tokens)

    for sentence in second_sentence:
        tokens = nltk.word_tokenize(sentence)
        second_sentence_tokens.insert(len(second_sentence_tokens), tokens)
        second_sentence_tags.insert(
            len(second_sentence_tags), nltk.pos_tag(tokens))

        # print(second_sentence_tokens)

    # lemmatizing
    first_sentence_lemmas = []
    second_sentence_lemmas = []
    lemmatizer = WordNetLemmatizer()
    for sentence in first_sentence_tokens:
        sentence_components = []
        for token in sentence:
            lemmas = lemmatizer.lemmatize(token)
            sentence_components.insert(len(sentence_components), lemmas)
        first_sentence_lemmas.insert(
            len(first_sentence_lemmas), sentence_components)

    for sentence in second_sentence_tokens:
        sentence_components = []
        for token in sentence:
            lemmas = lemmatizer.lemmatize(token)
            sentence_components.insert(len(sentence_components), lemmas)
        second_sentence_lemmas.insert(
            len(second_sentence_lemmas), sentence_components)

    return first_sentence_tokens, second_sentence_tokens


first_sentence, second_sentence, score = readData()


# EVERYTHING ABOVE HERE IS SETUP

# deterimine the cosine similarity between two sentences (feature for the ML model)
# REF: http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
def calc_cosine_similarity(sentence1, sentence2):

    # remove the stopwords, transform into TF-IDF matrix, then
    tfidf_matrix = TfidfVectorizer(
        stop_words="english").fit_transform([sentence1, sentence2])
    cos_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    cos_sim = cos_sim_matrix[0][1]

    return cos_sim


cos_sim = calc_cosine_similarity(first_sentence[1480], second_sentence[1480])
print('score', cos_sim)
