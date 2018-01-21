# all references to datasets are stored outside of this script
# in order to remove any unecessarily large git commits

# throughout this script, I will reference the dataset that was provided
# via GroupLens - a research lab at the University of Minnesota, Twin Cities
# which specializes in recommender systems, online communities mobile and
# ubiquitous, digital libraries, and local geographic information systems.

# Their mission statement is:

# We advance the theory and practice of social computing by 
# building and understanding systems used by real people.

# I have also integrated guidance from the nltk book http://www.nltk.org/book/ch06.html
# whereas, the book walks through the identification of gender based on trained names,
# I will be classifying movie reviews using similar methods.

# ~~~
# Data description:
# the directory structure that I am referencing is organized in the following manner:
#	- data
#		- positive reviews
#		- negative reviews

# I will train based off this structure

# ~~~

# Script objective: implement a basic Naive Bayes Classifier to identify the
# sentiment associated with particular words

# Define import set
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
# import csv # I shouldn't need this since I'm working with txt files
import os
import time
import unicodedata
import string
import re
import csv

data_store = '/Volumes/Jameson/github/data_store/review_polarity_training/'
punctuation_removal = set(string.punctuation)

def feature_attribution(corpus, feature):
	# generate lists of statements using the sentence chunker
	# to be able to easily call a single word generated within
	# the naive bayes counter dict process
	# to them be able to review examples of usage
	feature_examples = []
	sentence_chunks = sent_tokenize(corpus)
	for sent in sentence_chunks:
		word_tokens = set(word_tokenize(sent))
		if feature in word_tokens:
			feature_examples.append(sent)

	return feature_examples

def read_reviews(loc):
	# exports as a flat list
	print('processing: ' + loc)
	file_list = [f for f in os.listdir(loc) if not f.startswith('.')] # hidden files are annoying...
	corpus_master = []
	for f in file_list:
		temp_corpus = open(loc + f, 'r', errors='ignore', encoding='utf8')
		temp_reviews = temp_corpus.read().strip()
		corpus_master.append(temp_reviews)

	return corpus_master

def counter_dict(corpus):
	stop_words = set(stopwords.words('english'))
	word_count = {}
	sentence_count = {}
	callout_dict = {}

	for review in corpus:
		temp_tokens = word_tokenize(review)
		tokens_cleaned = [word for word in temp_tokens if not word in punctuation_removal and not word in stop_words]
		# count the number of sentences that a given token appears in across a corpus
		for token in set(tokens_cleaned):
			if token not in sentence_count:
				sentence_count[token] = 1
			else:
				sentence_count[token] += 1

		# count the number of times that a token appears across a review
		for word in tokens_cleaned:
			if word not in word_count:
				word_count[word] = 1
			else:
				word_count[word] += 1

	# ignore all items that only occur once
	{k: v for k, v in word_count.items() if v > 1}
	{k: v for k, v in sentence_count.items() if v > 1}

	return word_count, sentence_count

def compute_sentiment(pos_dict, neg_dict):
	# consider that words with count <= 1 are excluded from both
	print('computing overall sentiment classification...')
	naive_bayes_classifier = {}
	pos_keys = pos_dict[0].keys()
	neg_keys = neg_dict[0].keys()

	overlap = [word for word in pos_keys if word in neg_keys]
	print('pos file contains: ' + str(len(pos_dict[0].keys())) + ' keys')
	print('neg file contains: ' + str(len(neg_dict[0].keys())) + ' keys')
	
	for word in overlap:
		# 2 calculations: pos probability and neg probability
		# sent_prob = (sentence likelihood %) * (word likelihood %)
		pos_prob = float(pow(10, 8)) * float((pos_dict[0][word] / len(pos_keys))) * float((pos_dict[1][word] / len(pos_keys)))
		neg_prob = float(pow(10, 8)) * float((neg_dict[0][word] / len(neg_keys))) * float((neg_dict[1][word] / len(neg_keys)))
		if pos_prob > neg_prob:
			naive_bayes_classifier[word] = [word, 'positive', pos_prob, neg_prob, abs(pos_prob - neg_prob)]
		elif pos_prob < neg_prob:
			naive_bayes_classifier[word] = [word, 'negative', pos_prob, neg_prob, abs(pos_prob - neg_prob)]
		else:
			naive_bayes_classifier[word] = [word, 'unknown', pos_prob, neg_prob, abs(pos_prob - neg_prob)]

	return naive_bayes_classifier


def main():
	# compute positive statistics
	positive_reviews = read_reviews(data_store + 'pos/')
	pos_count_info = counter_dict(positive_reviews)

	# compute negative statistics
	negative_reviews = read_reviews(data_store + 'neg/')
	neg_count_info = counter_dict(negative_reviews)

	# compute sentiment statistics
	nb_classifier = compute_sentiment(pos_count_info, neg_count_info)
	
	# for status in nb_classifier:
	# 	print(status + ': ' + str(nb_classifier[status]))
	
	## from the above, I notice that the word 'produced' has a severly
	## negative rating; I will parse for example from both to identify what
	## is happening
	pos_ex = feature_attribution(''.join(positive_reviews), 'produced')
	neg_ex = feature_attribution(''.join(negative_reviews), 'produced')

	# for row in nb_classifier:
	# 	print(str(row) + str(nb_classifier[row]))

	with open(data_store + 'nb_output.csv', 'w') as csv_out:
		csv_writer = csv.writer(csv_out, delimiter=',')
		for row in nb_classifier:
			csv_writer.writerow(nb_classifier[row])

	# print('Printing positive examples:')
	# for x in pos_ex:
	# 	for y in pos_ex[x]:
	# 		print(y)
	# 		print('\n')

	# print('Printing negative examples:')
	# for x in neg_ex:
	# 	for y in neg_ex[x]:
	# 		print(y)
	# 		print('\n')

	print('\n')
	print('complete')

main()




















