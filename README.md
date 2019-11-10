# Semantic Textual Similarity

## Features

- Cosine Similarity -- bag of words style feature using TF-IDF

## Notes from Pat's 11/3 Work Session

The lemmatizer looks like it's doing a few weird things. For example in line 1 sentence 1, it gives the following:

```
['but', 'other', 'source', 'close', 'to', 'the', 'sale', 'said', 'vivendi', 'wa', 'keeping', 'the', 'door', 'open', 'to', 'further', 'bid', 'and', 'hoped', 'to', 'see', 'bidder', 'interested', 'in', 'individual', 'asset', 'team', 'up']
```

It seems that hoped, interested, bidder should be changed?

### Good link!

https://www.geeksforgeeks.org/get-synonymsantonyms-nltk-wordnet-python/

### Stanford CoreNLP Setup

To setup the stanford-corenlp, go to this link and download the ~300mb core-nlp server:

https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

Unzip it in the main directory and run the starting script to get the server running. You can then uncomment the code to print the tree.
