from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math


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


first_sentence, second_sentence, score = readData()

corpus = first_sentence+second_sentence

vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf = vect.fit_transform(corpus)
pairwise_similarity = tfidf * tfidf.T

arr = pairwise_similarity.toarray()
np.fill_diagonal(arr, np.nan)

# print('corpus[0:10]', corpus[0:10])

# input_doc = "the support will come as a free software upgrade called webvpn for current customers that have support contracts"

# get the index of the input sentence
# input_idx = corpus.index(input_doc)

# determine based on the tfid which sentence is most like this one
# result_idx = np.nanargmax(arr[input_idx])
# print(corpus[result_idx])


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

    return tfid_similarity


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

        print(true_score, tfid_score, tfid_score_rounded,
              true_score == tfid_score_rounded)
        # calculate the unrounded error
        line_error = abs(true_score - tfid_score)
        error += line_error
    print('average error: ', error / lines)
    print('percent error: ', correct / lines)


test()
