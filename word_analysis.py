"""
Briefing: The code is an implementation of a WordNetwork class in Python that performs several NLP tasks including:
          > Text cleaning
          > Generation of n-grams
          > Word frequency analysis
          > Wordcloud visualization
          > Topic modeling
          > General text statistics

  It starts by processing the input data and stopwords, then it calls several methods to perform each of the
  tasks mentioned above. The results of these tasks are then printed or visualized, including a graph
  representation of the co-occurrence matrix, the top n-grams, word frequency histogram, entropy of the text,
  etc.
"""

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


class TextAnalysis:
    def __init__(self,
                 input_filename: str,
                 stopword_filename: str):
        self.input_filename = input_filename
        self.input_filepath = 'Inputs/{}'.format(input_filename)
        self.stopword_filepath = 'Inputs/{}'.format(stopword_filename)

        self.raw_words = []
        self.stop_words = []
        self.filtered_words = []
        self.vocabulary = []
        self.ngrams = []
        self.top_comm = []
        self.text_entropy = None

        self.min_word_length = 2
        self.word_window = 4
        self.n_size = 2
        self.num_similar = 10
        self.num_topics = 5

        # updating the stopword list
        self.additional_stopwords = ['engineering', 'data', 'engineers', 'digital', 'today', 'use', 'used']

        self._preprocess_data()
        self._visualize_adjacency_matrix()
        self._extract_unique_words(self.raw_words)
        self._generate_ngrams()
        self._calculate_frequency()
        self._print_most_frequent_ngrams()
        self._generate_wordcloud()
        self._topic_modeling()
        self._calculate_entropy()
        self._print_text_statistics()

    def _preprocess_data(self):
        print(f"\n---This is an analysis for: {self.input_filepath}")

        self.raw_words = read_input(file=self.input_filepath)
        self.stop_words = read_input(file=self.stopword_filepath)
        self.stop_words.extend(self.additional_stopwords)

    def _visualize_adjacency_matrix(self) -> None:
        adj_matrix = self._co_occurrence()
        # print (adj_matrix)

        G = nx.from_pandas_adjacency(adj_matrix)
        visual_graph = Network(
            height="500px",
            bgcolor="#222222",
            font_color="white"
        )
        visual_graph.from_nx(G)
        visual_graph.show_buttons(filter_=['physics'])
        visual_graph.save_graph(self.input_filename + '.html')

    def _extract_unique_words(self, words):
        self.filtered_words = []
        self.vocabulary = set()

        for line in words:
            parts = line.strip().split()
            for _word in parts:
                word = _word.lower()
                if word not in self.stop_words and word.isalpha() and len(word) > self.min_word_length:
                    self.filtered_words.append(word)
                    self.vocabulary.add(word)

    def _generate_ngrams(self) -> None:
        self.ngrams = ['_'.join(self.filtered_words[i:i + self.n_size]) for i in
                       range(len(self.filtered_words) - self.n_size + 1)]

    def _calculate_frequency(self) -> None:
        self.top_comm = [ngram for ngram, count in Counter(self.ngrams).most_common(self.num_similar)]

    def _print_most_frequent_ngrams(self) -> None:
        top_ngrams = ["{}. {}".format(_id + 1, ngram) for _id, ngram in enumerate(self.top_comm)]
        print('\nThe following are the top {} -grams:\n{}'.format(
            self.num_similar,
            '\n'.join(top_ngrams)))

    def _generate_wordcloud(self) -> None:
        all_words_string = ' '.join(self.filtered_words + self.top_comm)
        all_stopwords_string = set(' '.join(self.stop_words))

        # defining the wordcloud parameters
        wc = WordCloud(background_color="white",
                       max_words=2000,
                       stopwords=all_stopwords_string)

        # generate words cloud
        wc.generate(all_words_string)

        # store to file
        _file = 'Outputs/{}.{}'.format(self.input_filename, '.png')
        wc.to_file(filename=_file)

        # show the cloud
        plt.imshow(wc)
        plt.axis('off')
        plt.show()

    def _topic_modeling(self):
        tokens = [x.split() for x in self.filtered_words]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(text) for text in tokens]

        lda = LdaModel(corpus=corpus,
                       num_topics=self.num_topics,
                       id2word=dictionary)

        print('\n the following are the top', self.num_topics, 'topics:')

        for i in range(0, len(lda.show_topics(self.num_topics))):
            print(lda.show_topics(self.num_topics)[i][1], '\n')

        lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

        return pyLDAvis.display(lda_display)

    def _co_occurrence(self):
        d = defaultdict(int)
        vocab = set()
        for text in self.raw_words:
            text = text.lower().split()
            self._extract_unique_words(text)
            for i in range(len(self.filtered_words)):
                token = self.filtered_words[i]
                vocab.add(token)
                next_token = self.filtered_words[i + 1: i + 1 + self.word_window]
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

    def _calculate_entropy(self):
        words_counter = Counter(self.filtered_words)
        word_freq_lst = list(words_counter.values())
        self.text_entropy = entropy(word_freq_lst, base=10)

    def _print_text_statistics(self):
        stats = {
            'total_words':   len(self.filtered_words),
            'unique_words':  len(self.vocabulary),
            'total_entropy': round(log(self.text_entropy, 10), 2)
        }

        message = '\nThe following are some basic statistics on the input text.' \
                  '\n\tTotal number of words: {total_words}' \
                  '\n\tTotal number of unique words: {unique_words}' \
                  '\n\tTotal entropy in the text: {total_entropy}'.format(**stats)

        print(message)


def read_input(file):
    file_name = file if file.lower().endswith('.txt') else file + '.txt'

    try:
        with open(file=file_name, mode='r', encoding='utf8') as fr1:
            return [word.strip() for word in fr1 if word]
    except FileNotFoundError:
        print('Error reading the input data')


if __name__ == '__main__':
    try:
        TextAnalysis(input_filename='Robert_Liu.txt',  # Provide the input text file here
                     stopword_filename='stopwords_en.txt')  # Provide the stopword file here
    except Exception as ex1:
        print('Something went wrong during execution: {}'.format(ex1))
