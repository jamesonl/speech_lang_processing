import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
from nltk.corpus import stopwords
import string

def n_gram_gen(gram_len, sent_split):
	# pop and insert sentence start and end markers relative to length
	# of gram_len
	# print( 'running n_gram tokenizer')
	n_gram = {}
	for sent in sent_split:
		sent = sent.translate(string.punctuation)
		word_tokens = nltk.word_tokenize(sent)
		n = 0

		while n < gram_len:
			word_tokens.insert(0, '<s>')
			word_tokens.append('</s>')
			n += 1

		# credit to Scott Triglia for this elegant solution
		gram = zip(*[word_tokens[i:] for i in range(gram_len)])
		for g in gram:		
			if g not in n_gram:
				n_gram[' '.join(g)] = 1
			else:
				n_gram[' '.join(g)] += 1
		
	return n_gram

def relative_frequencies(ngram_len, model):
	# create my own Markov Chain to identify relative frequencies
	# relative frequencies represent the probabilities given the chain rule of probability
	# i.e. the join probability of a squence & computing their conditional probability

	# everything is split by spaces within my models, so I can resplit the keys
	# in order to get a list of all the different combos
	rel_freq = {}
	last_w = {}
	leading_phrase = {}
	keys = model.keys()
	for k in keys:
		words = k.split(' ')

		# two dictionaries for reference
		gram = words[0:ngram_len-1]

		# this will track grams related to a particular word
		if words[ngram_len-1] not in last_w:
			last_w[(words[ngram_len-1])] = [' '.join(gram)]
		else:
			last_w[(words[ngram_len-1])].append(' '.join(gram))

		# this will track phrases start with the leading phrase
		if ' '.join(gram) not in leading_phrase:
			leading_phrase[' '.join(gram)] = [' '.join(words)]
		else:
			leading_phrase[' '.join(gram)].append(' '.join(words))

	# relative frequency is calculated by dividing:
	# the number of times a word appears following a phrase, by
	# the number of times that phrase appears within the corpus
	for k in last_w:
		for phrase in last_w[k]:
			phrase_usage = float(len(leading_phrase[phrase]))
			occurrence = float(last_w[k].count(phrase))
			rel_freq[str(k)] = float(occurrence / phrase_usage)

	return rel_freq, leading_phrase


def calc_perplexity(model, rel_freq_ngram_dict):
	# given the probabilities of each word across a corpus,
	# the perplexity is a inverse measure of how likely a word is to occur
	# given a set of phrases
	perplexity = 1
	N = 0

	for ph in model.keys():
		words = ph.split(' ')
		prob = 1
		for w in words:
			prob *= rel_freq_ngram_dict[w] 
			N += 1
			# print( w, (1 / rel_freq_ngram_dict[w]), perplexity)
	perplexity = pow(1 / prob, 1 / float(N))

	return perplexity


def main():
	print( 'loading corpus')
	corpus = open('pride_prejudice.txt', encoding='latin-1')
	stop_words = set(stopwords.words('english'))

	print( 'tokenizing sentences')
	sentence_split = nltk.sent_tokenize(corpus.read())
	
	print( 'starting bigram generator')
	unigram = n_gram_gen(1, sentence_split)
	bigram = n_gram_gen(2, sentence_split)
	trigram = n_gram_gen(3, sentence_split)
	quadgram = n_gram_gen(4, sentence_split)

	# generate the relative frequencies of every model
	rel_bigrams = relative_frequencies(2, bigram)
	rel_trigrams = relative_frequencies(3, trigram)
	rel_quadgrams = relative_frequencies(4, quadgram)
	# rel_freqs = rel_trigrams[0]
	# rel_comps = rel_trigrams[1]

	# SANITY CHECK
	# rel_quads = relative_frequencies(4, quadgram)
	# rel_freqs = rel_quads[0]
	# rel_comps = rel_quads[1]

	# for x in rel_freqs:
	# 	print( x, rel_freqs[x], rel_comps[x.split('|')[1]])

	# GENERATE THE PERPLEXITY OF EACH MODEL
	bigrams_perplexity = calc_perplexity(bigram, rel_bigrams[0])
	trigrams_perplexity = calc_perplexity(trigram, rel_trigrams[0])
	quadgrams_perplexity = calc_perplexity(quadgram, rel_quadgrams[0])
	
	print( 'Bigram Perplexity: ' + str(bigrams_perplexity))
	print( '\n')

	print( 'Trigrams are a better model by a factor of: ' + str(round(bigrams_perplexity / trigrams_perplexity, 4)))
	print( 'Trigram Perplexity: ' + str(trigrams_perplexity))

	print( '\n')
	print( 'Quadgrams are a better model by a factor of: ' + str(round(trigrams_perplexity / quadgrams_perplexity, 4)))
	print( 'Quadgram Perplexity: ' + str(quadgrams_perplexity))

main()