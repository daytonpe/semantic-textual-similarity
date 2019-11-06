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


# Calculate the freqency distribution for a corpus
def frequency_distribution():
    freq_dist = FreqDist()
    for i in range(len(first_sentence)):
        for token in (first_sentence_tokens[i] + second_sentence_tokens[i]):
            freq_dist[token.lower()] += 1
    # print(freq_dist.most_common(40))
    return freq_dist


def run_tfid(sentence1, sentence2, true_score):

    # print('sentence 1:', sentence1)
    # print('sentence 2:', sentence2)
    # print('score', true_score)
    tfidf1 = TfidfVectorizer(
        min_df=1, stop_words="english").fit_transform([sentence1, sentence2])
    pairwise_similarity1 = tfidf1 * tfidf1.T
    # print('pairwise_similarity1\n', pairwise_similarity1)

    tfid_similarity = (pairwise_similarity1.toarray()[0][1]*5)

    # print('TFID similarity: ', tfid_similarity)
    # this should be the cosine similarity if i did it right
    # https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
    # it's bag of words tho so it's pretty inaccurate.
    return tfid_similarity


first_sentence, second_sentence, score = readData()

corpus = first_sentence+second_sentence

vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf = vect.fit_transform(corpus)
pairwise_similarity = tfidf * tfidf.T

arr = pairwise_similarity.toarray()
np.fill_diagonal(arr, np.nan)


first_sentence_tokens, second_sentence_tokens = preprocess()
freq_dist = frequency_distribution()


def run_sif(sentence1, sentence2, true_score):
    a = .001
    for word in sentence1:
        w = a / (a + freq_dist[word])
        print(w)
        # print(vect.get_feature_names())
        # print(corpus[:10])

        # corpus = first_sentence+second_sentence
        # vectorizer = TfidfVectorizer()
        # X = vectorizer.fit_transform(corpus)
        # # print(vectorizer.get_feature_names()) # all the unique words in corpus
        # print(X.shape)

        # run_tfid(first_sentence[0], second_sentence[0], score[0])


def test():
    error = 0
    correct = 0
    lines = len(first_sentence)
    for i in range(0, lines):
        true_score = int(score[i])
        tfid_score = run_tfid(first_sentence[i], second_sentence[i], score[i])

        # calculate the percentage of correct guesses
        tfid_score_rounded = round(tfid_score)
        if tfid_score == true_score:
            correct += 1

        # print(true_score, tfid_score, tfid_score_rounded,
        #       true_score == tfid_score_rounded)
        # calculate the unrounded error
        line_error = abs(true_score - tfid_score)
        error += line_error
    print('average error: ', error / lines)
    print('percent error: ', correct / lines)


run_sif(first_sentence[0], second_sentence[0], score[0])
