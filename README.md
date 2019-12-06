# Sentence Similarity with Random Forest

## UTD CS6320 - Natural Language Processing - Fall 2019

### Shruti Agrawal and Pat Dayton

## Description of the Problem Statement

Our task was to design and implement a model for determining the textual similarity between two sentences or chunks of text on a 1 to 5 scale. A higher score means that the two sentences are more similar. A lower score means they are less similar. Some examples from the training data are seen here:

Sentence 1: ''The Hulk'' was a monster at the box office in its debut weekend, taking in a June opening record of $62.6 million.	
Sentence 2:  "The Hulk" took in $62.6 million at the box office, a monster opening and a new June record.
Score: 5

Sentence 1: Congratulations on being named Time magazine's Person of the Year.
Sentence 2: Time magazine named the American soldier its Person of the Year for 2003.
Score: 2

## Possible Use Cases for these Techniques

Textual similarity, at its root, is simply a metric for determining the meaning of a given chunk of text with respect to another corpus. Thus there are myriad uses for the technology.
Search engines like Google use text similarity to ensure their users are receiving links to sites with a high degree of similarity to the search terms provided. Online forums, specifically question and answer based forums, need a way to identify duplicate questions so that they can be grouped and more easily be found by their users. Quora, an online Q&A forum company, created a Kaggle competition to help improve their algorithm.

## Approach

Our proposed solution was to create a variety of features describing each sentence in terms of its word choice and semantic properties. These features were then fed into a variety of decision-tree-based models including SciKit Learn’s Decision Tree and Random Forest Models, and the more powerful ensemble machine learning model called XGBoost (more later). Python was used for the application.
