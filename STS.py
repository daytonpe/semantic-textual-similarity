# -*- tfoding: utf-8 -*-
"""
Created on Sat Oct 26 20:07:07 2019

@author: Shruti Agrawal & Pat Dayton
"""


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.corenlp import CoreNLPParser
from nltk.corpus import wordnet as wn

core_nlp_url = 'http://localhost:9000'


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

    # Some print statements to check outputs
    print("\nPrinting the first sentence: ")
    print(first_sentence[0])
    print("\nPrinting the tokens of first sentence: ")
    print(first_sentence_tokens[0])
    print("\nPrinting the POS tags of the first sentence: ")
    print(first_sentence_tags[0])
    print("\nPrinting the lemmas of the first sentence: ")
    print(first_sentence_lemmas[0])

    # dependency parsing
    print("\nDependency parsing: ")
    # dependency_parser = CoreNLPDependencyParser(url=core_nlp_url)
    # parse, = dependency_parser.raw_parse(first_sentence[0])
    # print(parse.to_conll(4))

    # syntactic parsing
    print("\nFull syntactic parse tree: ")
    # syntactic_parser = CoreNLPParser(url=core_nlp_url)
    # next(syntactic_parser.raw_parse(first_sentence[0])).pretty_print()

    # WORDNET connections. Should only be made for Nouns, Verbs,

    print("\n WORDNET: ")
    for w, pos in zip(first_sentence_tokens[0], first_sentence_tags[0]):
        synonyms = []
        hypernyms = []
        hyponyms = []
        substance_meronyms = []
        part_meronyms = []
        holonyms = []

        print('\n', pos[0], pos[1])

        for syn in wn.synsets(w):
            # Synonyms
            for l in syn.lemmas():
                if l.name() not in synonyms:
                    synonyms.append(l.name())

            # Hypernyms
            for hpr in syn.hypernyms():
                for l in hpr.lemmas():
                    if l.name() not in hypernyms:
                        hypernyms.append(l.name())

            # Hyponyms
            for hpo in syn.hyponyms():
                for l in hpo.lemmas():
                    if l.name() not in hyponyms:
                        hyponyms.append(l.name())

            # Substance Meronyms
            for mrn in syn.substance_meronyms():
                for l in mrn.lemmas():
                    if l.name() not in substance_meronyms:
                        substance_meronyms.append(l.name())

            # Part Meronyms
            for mrn in syn.part_meronyms():
                for l in mrn.lemmas():
                    if l.name() not in part_meronyms:
                        part_meronyms.append(l.name())

            # Holonyms
            for hol in syn.member_holonyms():
                for l in hol.lemmas():
                    if l.name() not in holonyms:
                        holonyms.append(l.name())

        print('Synonyms: ', synonyms)
        print('Hypernyms: ', hypernyms)
        print('Hyponyms: ', hyponyms)
        print('Meronyms (substance): ', substance_meronyms)
        print('Meronyms (part): ', part_meronyms)
        print('Holonyms:', holonyms)


if __name__ == '__main__':
    preprocess()
