from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.corenlp import CoreNLPParser
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
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


# Calculate the freqency distribution for a corpus
def frequency_distribution():
    freq_dist = FreqDist()
    for i in range(len(first_sentence)):
        for token in (first_sentence_tokens[i] + second_sentence_tokens[i]):
            freq_dist[token.lower()] += 1
    # print(freq_dist.most_common(40))
    return freq_dist


first_sentence, second_sentence, score = readData()
first_sentence_tokens, second_sentence_tokens = preprocess()
corpus = first_sentence+second_sentence
freq_dist = frequency_distribution()


def run_sif(sentence1, sentence2, a=.001):
    tfidf_vector = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vector.fit_transform([sentence1, sentence2])
    print('tfidf_matrix', tfidf_matrix)
    print(tfidf_vector.get_feature_names())
    print(tfidf_vector.get_params())

    for word in sentence1:
        w = a / (a + freq_dist[word])
        # print(w)
        # print(vect.get_feature_names())
        # print(corpus[:10])

        # corpus = first_sentence+second_sentence
        # vectorizer = TfidfVectorizer()
        # X = vectorizer.fit_transform(corpus)
        # # print(vectorizer.get_feature_names()) # all the unique words in corpus
        # print(X.shape)

        # run_tfid(first_sentence[0], second_sentence[0], score[0])


# run_sif('I like some apples', 'he loves the apples')
corpus = ['I like some apples', 'he loves the red apples that are red']
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())
