from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import pyLDAvis.gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from pyvis.network import Network


class TextAnalysis:
    def __init__(self,
                 input_filename: str,
                 stopword_filename: str):
        self.input_filename = input_filename
        self.word_window = None
        self.input_filepath = '{}'.format(input_filename)
        self.stopword_filepath = '{}'.format(stopword_filename)

        self.raw_words = read_input(file=self.input_filepath)
        self.stop_words = read_input(file=self.stopword_filepath)
        self.additional_stopwords = ['engineering', 'data', 'engineers', 'digital', 'today', 'use', 'used']
        self.stop_words.extend(self.additional_stopwords)

        self.filtered_words = []
        self.vocabulary = []

        self.min_word_length = 2
        self.num_topics = 5

        self._extract_unique_words(self.raw_words)
        self._topic_modeling()
        self._visualize_adjacency_matrix()

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

    def _topic_modeling(self):
        tokens = [x.split() for x in self.filtered_words]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(text) for text in tokens]

        lda = LdaModel(corpus=corpus,
                       num_topics=self.num_topics,
                       id2word=dictionary)

        # print('\n the following are the top', num_topics, 'topics:')

        for i in range(0, len(lda.show_topics(self.num_topics))):
            print(lda.show_topics(self.num_topics)[i][1], '\n')

        lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

        return pyLDAvis.display(lda_display)

    def co_occurrence(self):
        d = defaultdict(int)
        vocab = set()
        for text in self.raw_words:
            text = text.lower().split()
            self._extract_unique_words(text)
            for i in range(len(self.filtered_words)):
                token = self.filtered_words[i]
                vocab.add(token)  # add to vocab
                next_token = self.filtered_words[i + 1: i + 1 + self.word_window]
                for t in next_token:
                    key = tuple(sorted([t, token]))
                    d[key] += 1

        # sort vocab
        vocab = sorted(vocab)

        # formulate the dictionary into dataframe
        df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                          index=vocab,
                          columns=vocab)

        for key, value in d.items():
            df.at[key[0], key[1]] = value
            df.at[key[1], key[0]] = value

        return df

    def _visualize_adjacency_matrix(self) -> None:
        adj_matrix = self.co_occurrence()
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


def read_input(file):
    file_name = file if file.lower().endswith('.txt') else file + '.txt'
    with open(file=file_name, mode='r', encoding='utf8') as fr1:
        return [word.strip() for word in fr1 if word]


if __name__ == '__main__':
    TextAnalysis(input_filename='Robert_Liu.txt',  # Provide the input text file here
                 stopword_filename='stopwords_en.txt')  # Provide the stopword file here)
