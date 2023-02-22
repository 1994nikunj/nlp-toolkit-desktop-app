from collections import Counter, defaultdict
from math import log

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyLDAvis.gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from pyvis.network import Network
from scipy.stats import entropy
from wordcloud import WordCloud


# cleans text
def txt_clean(word_list, stopwords_list, min_len):
    _clean_words = []
    vocab = []
    for line in word_list:
        parts = line.strip().split()
        for _word in parts:
            word_l = _word.lower()
            # print (word_l, '\n', '\n')
            if word_l not in stopwords_list:
                if word_l.isalpha():
                    if len(word_l) > min_len:
                        _clean_words.append(word_l)
                        if word_l not in vocab:
                            vocab.append(word_l)
    return _clean_words, vocab


# creates a list of the "num" most common elements/strings in an input list
def most_common(lst, num):
    data = Counter(lst)
    common = data.most_common(num)
    top_comm = []
    for i in range(0, num):
        top_comm.append(common[i][0])
    return top_comm


# creates a list of n-grams. Individual words are joined together by "_"
def ngram(text, grams):
    n_grams_list = []
    count = 0
    for _token in text[:len(text) - grams + 1]:
        n_grams_list.append(text[count] + '_' + text[count + grams - 1])
        count += 1
    return n_grams_list


# creates a matrix of co-occurrence
def co_occurrence(sentences, window_size, stop_list, len_w):
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        text = text.lower().split()
        text, vocab2 = txt_clean(text, stop_list, len_w)
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)
            next_token = text[i + 1: i + 1 + window_size]
            for t in next_token:
                key = tuple(sorted([t, token]))
                d[key] += 1
    vocab = sorted(vocab)
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df


# creating and presenting the wordcloud
def cloud(words, stpwords, file_name):
    # Defining the wordcloud parameters
    wc = WordCloud(background_color="white", max_words=2000,
                   stopwords=stpwords)
    # Generate words cloud
    wc.generate(words)
    # Store to file
    wc.to_file(file_name + '.png')
    # Show the cloud
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


# extracting the topics using LDA
def topic_modeling(words, num_of_topics, visualize=False):
    tokens = [x.split() for x in words]
    # print (words)
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]

    lda = LdaModel(corpus=corpus, num_topics=num_of_topics, id2word=dictionary)
    print('\nthe following are the top', num_topics, 'topics:')
    for i in range(0, len(lda.show_topics(num_of_topics))):
        print(lda.show_topics(num_of_topics)[i][1], '\n')

    if visualize:
        lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
        return pyLDAvis.display(lda_display)


# reading input files
# input_file_name = 'Anthony_Farmand'
input_file_name = 'Robert_Liu'
txt_file = open(input_file_name + '.txt', 'r', encoding='utf8')
stopwords_file = open('stopwords_en.txt', 'r', encoding='utf8')

# initializing lists
stopwords = []
txt_words = []

# populating the list of stopwords and the list of words from the text file
for word in stopwords_file:
    stopwords.append(word.strip())

for word in txt_file:
    txt_words.append(word.strip())

# updating the stopword list
# Anthony_Farmand_stopwords = ['engineering', 'data', 'engineers', 'digital', 'person', 'civil', 'san']
# stopwords.extend(Anthony_Farmand_stopwords)
Robert_Liu_stopwords = ['engineering', 'data', 'engineers', 'digital', 'today', 'use', 'used']
stopwords.extend(Robert_Liu_stopwords)

# setting the minimum word lenght
min_word_len = 2

# setting the window of separation between words
word_window = 4

# setting the "n" for the n-grams generated
n_value = 2

# setting the number of elements to be similar
n_sim_elems = 10

# defining the name of the file for the wordcloud (no extension required)
cloud_filename = input_file_name

# defining the number of topics to be extracted
num_topics = 5

# generating the co-occurence matrix and extracting a graph from the resulting adjacency matrix
adj_matrix = co_occurrence(txt_words, word_window, stopwords, min_word_len)
# print (adj_matrix)
G = nx.from_pandas_adjacency(adj_matrix)
# visualizing the network
G_viz = Network(height="500px", bgcolor="#222222", font_color="white")
G_viz.from_nx(G)
G_viz.show_buttons(filter_=['physics'])
G_viz.show(input_file_name + '.html')

# cleaning the words and getting the list of unique words
clean_words, vocabulary = txt_clean(txt_words, stopwords, min_word_len)

# generating the n-grams and taking the top "n" elements (equal to "n_sim_elems")
ngrams_list = ngram(clean_words, n_value)
top_ngrams = most_common(ngrams_list, n_sim_elems)
print('\n---This is an analysis for', input_file_name)
print('\nthe following are the top', n_sim_elems, '-grams:\n', top_ngrams, '\n')

# generating the wordcloud
#   creating a string of words and top ngrams
all_words_string = ' '.join(clean_words + top_ngrams)
all_stopwords_string = ' '.join(stopwords)
#   calling wordcloud generator
cloud(all_words_string, all_stopwords_string, cloud_filename)

# extracting topics
topic_modeling(clean_words, num_topics)

# general statistics on the input text
tot_num_words = len(clean_words)
unique_words_num = len(vocabulary)
# calculating the entropy in the text
words_counter = Counter(clean_words)
word_freq_lst = list(words_counter.values())
entropy_val = entropy(word_freq_lst, base=10)

print('\nthe following are some basic statistics on the input text')
print('   total number of words:', tot_num_words)
print('   total number of unique words:', unique_words_num)
print('   total entropy in the text:', log(entropy_val, 10), '\n', '     (entropy is a measure of information rate)')

print('\n----this ends the process----\n')
