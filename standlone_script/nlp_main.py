import os
import warnings
from collections import Counter, defaultdict
from math import log

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyLDAvis.gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from pandas import DataFrame
from pyvis.network import Network
from scipy.stats import entropy
from wordcloud import WordCloud

warnings.filterwarnings("ignore", '.*the imp module.*')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='wordcloud')
warnings.filterwarnings("ignore", category=FutureWarning, module='pyLDAvis')


class TextAnalysis:
    def __init__(self, input_filename: str, stopword_filename: str):
        self.input_filename = input_filename
        self.stopword_filename = stopword_filename
        self.final_output = []

        self.raw_words = []
        self.stop_words = []
        self.filtered_words = []
        self.vocabulary = set()
        self.ngrams = []
        self.top_comm = []

        self.wordcloud = None
        self.lda = None

        self.text_entropy = None
        self.total_words = None
        self.unique_words = None

        self.n_size = 2
        self.num_topics = 5
        self.word_window = 4
        self.num_similar = 10
        self.min_word_length = 2
        self.top_ngrams = None

        # updating the stopword list
        self.additional_stopwords = ['engineering', 'data', 'engineers', 'digital', 'today', 'use', 'used']

        self._read_input_data()
        self._visualize_adjacency_matrix()
        self._text_cleaning(self.raw_words)
        self._generate_ngrams()
        self._calculate_frequency()
        self._print_most_frequent_ngrams()
        self._generate_wordcloud()
        self._topic_modeling()
        self._calculate_stats()
        self._print_text_statistics()
        self._write_to_output_file()

    def _read_input_data(self) -> None:
        self.final_output.append(f"This is an analysis for: {self.input_filename}")
        self.raw_words = read_input(file=self.input_filename)
        self.stop_words = read_input(file=self.stopword_filename)
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

        graph_name = '{}.{}'.format(self.input_filename.replace('.txt', ''), 'html')
        visual_graph.save_graph(graph_name)

    def _text_cleaning(self, words) -> None:
        # Split the modified text into individual words
        words = ' '.join(words).split()

        # Filter out stop words and non-alphabetic words
        filtered_words = [word.lower() for word in words if
                          word.lower() not in self.stop_words and word.isalpha() and len(word) > self.min_word_length]

        # Update the filtered words and vocabulary lists
        self.filtered_words = filtered_words
        self.vocabulary = set(filtered_words)

    def _generate_ngrams(self) -> None:
        self.ngrams = ['_'.join(self.filtered_words[i:i + self.n_size]) for i in
                       range(len(self.filtered_words) - self.n_size + 1)]

    def _calculate_frequency(self) -> None:
        self.top_comm = [ngram for ngram, count in Counter(self.ngrams).most_common(self.num_similar)]

    def _print_most_frequent_ngrams(self) -> None:
        self.top_ngrams = ["{}. {}".format(_id + 1, ngram) for _id, ngram in enumerate(self.top_comm)]
        self.final_output.append(
            '\nThe following are the top {}-grams: \n\t{}'.format(self.n_size, '\n\t'.join(self.top_ngrams)))

    def _generate_wordcloud(self) -> None:
        all_words_string = ' '.join(self.filtered_words + self.top_comm)
        all_stopwords_string = set(' '.join(self.stop_words))

        # defining the wordcloud parameters
        self.wordcloud = WordCloud(background_color="white",
                                   max_words=2000,
                                   stopwords=all_stopwords_string)

        # generate words cloud
        self.wordcloud_generated = self.wordcloud.generate(all_words_string)

        _file = '{}.{}'.format(self.input_filename.replace('.txt', ''), 'png')
        self.wordcloud.to_file(filename=_file)

        plt.imshow(self.wordcloud)
        plt.axis('off')
        plt.show()

    def _topic_modeling(self) -> None:
        tokens = [x.split() for x in self.filtered_words]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(text) for text in tokens]

        self.lda = LdaModel(corpus=corpus,
                            num_topics=self.num_topics,
                            id2word=dictionary)

        self.final_output.append(f'\nThe following are the top {self.num_topics} topics:')
        for i, topic in enumerate(self.lda.show_topics(num_topics=self.num_topics)):
            top_words = self.lda.show_topics(self.num_topics)[i][1]
            self.final_output.append(f"\tTopic {i + 1}: {top_words}")

        lda_display = pyLDAvis.gensim.prepare(self.lda, corpus, dictionary, sort_topics=False)

        return pyLDAvis.display(lda_display)

    def _co_occurrence(self) -> DataFrame:
        d = defaultdict(int)
        vocab = set()
        for text in self.raw_words:
            text = text.lower().split()
            self._text_cleaning(text)
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

    def _calculate_stats(self) -> None:
        words_counter = Counter(self.filtered_words)
        word_freq_lst = list(words_counter.values())
        self.text_entropy = entropy(word_freq_lst, base=10)

        self.total_words = len(self.filtered_words)
        self.unique_words = len(self.vocabulary)
        self.text_entropy = round(log(self.text_entropy, 10), 2)

    def _print_text_statistics(self) -> None:
        message = '\nThe following are some basic statistics on the input text:' \
                  f'\n\tTotal Words: {self.total_words}' \
                  f'\n\tTotal Unique words: {self.unique_words}' \
                  f'\n\tTotal Text Entropy: {self.text_entropy}'
        self.final_output.append(message)

    def _write_to_output_file(self):
        output_file = f'{os.getcwd()}\\output_{self.input_filename}'
        with open(output_file, 'w') as fr:
            fr.write('\n'.join(self.final_output))


def read_input(file) -> list[str]:
    file_name = file if file.lower().endswith('.txt') else file + '.txt'

    with open(file=file_name, mode='r', encoding='utf8') as fr1:
        return [word.strip() for word in fr1 if word]


if __name__ == '__main__':
    TextAnalysis(input_filename="Robert_Liu.txt", stopword_filename="stopwords_en.txt")
