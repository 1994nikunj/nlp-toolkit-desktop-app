"""
Briefing: The code is an implementation of a WordNetwork class in Python that performs several NLP tasks including:
          > Text cleaning
          > Generation of n-grams
          > Word frequency analysis
          > Wordcloud visualization
          > Topic modeling
          > General text statistics
          > Sentiment Analysis
          > Summarize Text

  It starts by processing the input data and stopwords, then it calls several methods to perform each of the
  tasks mentioned above. The results of these tasks are then printed or visualized, including a graph
  representation of the co-occurrence matrix, the top n-grams, word frequency histogram, entropy of the text,
  etc.
"""

try:
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
    from pyvis.network import Network
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
except ImportError():
    print('ImportError: Missing Imports')

nltk.download('vader_lexicon')


# ======================================================================================================================
class WordNetwork:
    def __init__(self,
                 input_file: str,
                 stop_file: str,
                 visualize: bool):

        # Local class variable declaration
        self.input_filename = input_file
        self.input_file = 'Inputs/{}'.format(input_file)
        self.stop_file = 'Inputs/{}'.format(stop_file)
        self.visualize = visualize

        self.top_comm = []
        self.txt_words = []
        self.vocabulary = []
        self.stop_words = []
        self.clean_words = []
        self.n_grams_list = []

        self.min_word_len = 2  # setting the minimum word length
        self.word_window = 4  # setting the window of separation between words
        self.n_value = 2  # setting the "n" for the n-grams generated
        self.n_sim_elems = 10  # setting the number of elements to be similar
        self.num_topics = 5  # defining the number of topics to be extracted

        # updating the stopword list
        self.robert_liu_stopwords = ['engineering', 'data', 'engineers', 'digital', 'today', 'use', 'used']

        print('\n---This is an analysis for: ', self.input_file)
        self.process_data()

        # cleaning the words and getting the list of unique words
        self.get_unique_words(self.txt_words, self.stop_words, self.min_word_len)

        # generating the n-grams and taking the top "n" elements (equal to "n_sim_elems")
        self.ngram()

        self.most_common()

        # visualizing the network
        if self.visualize:
            self.viz_adj_matrix()

        print('\nThe following are the top {} -grams:'.format(self.n_sim_elems))
        for _id, _ngram in enumerate(self.top_comm):
            print('{}. {}'.format(_id + 1, _ngram))

        # calling wordcloud generator
        self.word_cloud_generation()

        # extracting topics
        self.topic_modeling()

        # calculating the entropy in the text
        words_counter = Counter(self.clean_words)
        word_freq_lst = list(words_counter.values())
        entropy_val = entropy(word_freq_lst, base=10)

        print('\nThe following are some basic statistics on the input text.'
              '\n\tTotal Word Count: {total_words}'
              '\n\tTotal Unique Word Count: {unique_words}'
              '\n\tTotal Entropy in the text: {total_entropy}'.format(total_words=len(self.clean_words),
                                                                      unique_words=len(self.vocabulary),
                                                                      total_entropy=log(entropy_val, 10)))

    # ------------------------------------------------------------------------------------------------------------------
    # populating the list of stopwords and the list of words from the text file
    def process_data(self):
        self.txt_words = read_input(file=self.input_file)
        self.stop_words = read_input(file=self.stop_file)

        self.stop_words.extend(self.robert_liu_stopwords)

    # ------------------------------------------------------------------------------------------------------------------
    # generating the co-occurrence matrix and extracting a graph from the resulting adjacency matrix
    def viz_adj_matrix(self) -> None:
        """
        Generate a co-occurrence matrix and visualize the resulting adjacency matrix as a graph.

        The function first checks if the adjacency matrix is set to be generated. If yes, the function calls the
        co_occurrence method to generate the adjacency matrix.

        The adjacency matrix is then converted to a graph using NetworkX's from_pandas_adjacency method. The graph is
        then visualized using the networkx_phyloviz library's Network class, with specified height and background
        color. The visualized graph includes buttons for filtering and is saved to an HTML file with a specified
        filename.
        """
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
        visual_graph.show('/Outputs/' + self.input_filename + '.html')

    # ------------------------------------------------------------------------------------------------------------------
    # cleans text
    def get_unique_words(self, words, stopwords, length):
        """
        The function 'get_unique_words' takes three arguments:
            - words: list of words
            - stopwords: list of stopwords
            - length: minimum length of a word to be considered

        It returns two class attributes:
            - clean_words: list of words filtered from the original list 'words'
            - vocabulary: list of unique words from the filtered list 'clean_words'

        The function filters the words in the list 'words' by:
            - converting them to lowercase
            - removing stopwords
            - only considering words that are alphabetic and have a length greater than 'length'

        The filtered words are stored in the class attribute 'clean_words' and the unique words from this list are
        stored in the class attribute 'vocabulary'.
        """
        self.clean_words = []
        self.vocabulary = set()
        stopwords = set(stopwords)

        for line in words:
            parts = line.strip().split()
            for _word in parts:
                word = _word.lower()
                if word not in stopwords and word.isalpha() and len(word) > length:
                    self.clean_words.append(word)
                    self.vocabulary.add(word)

    # ------------------------------------------------------------------------------------------------------------------
    # creates a list of n-grams
    def ngram(self, join_type='_') -> None:
        """
        The ngram function creates a list of n-grams from the self.clean_words list. The n-grams are created by
        joining the individual words in the list using the join_type argument (default is '_').

        The function uses a list comprehension to iterate over the range of the self.clean_words list, taking a slice
        of self.n_value words at a time and joining them together using the join_type separator. The resulting
        n-grams are added to the self.n_grams_list list.

        The function updates the self.n_grams_list list in place.
        """
        self.n_grams_list = [join_type.join(self.clean_words[i:i + self.n_value]) for i in
                             range(len(self.clean_words) - self.n_value + 1)]

    # ------------------------------------------------------------------------------------------------------------------
    # creates a list of the "num" most common elements/strings in an input list
    def most_common(self) -> None:
        """
        The most_common method is used to get the n_sim_elems most common n-grams from the list self.n_grams_list.
        The method uses the Counter method from the collections module to calculate the frequency of each n-gram in
        the self.n_grams_list list.

        The result of Counter(self.n_grams_list) is a dictionary-like object that maps the n-grams to their
        frequency. The most_common method of this object returns a list of tuples, each tuple representing an n-gram
        and its frequency.

        The method then uses a list comprehension to extract the first element of each tuple (i.e., the n-gram) and
        store it in the list self.top_comm. The list comprehension iterates over the list most_common(self.n_sim_elems),
        which is the list of n_sim_elems most common n-grams and their frequencies.
        """
        self.top_comm = [elem[0] for elem in Counter(self.n_grams_list).most_common(self.n_sim_elems)]

    # ------------------------------------------------------------------------------------------------------------------
    # creating and presenting the wordcloud
    def word_cloud_generation(self) -> None:
        """
        Generates a word cloud from a list of words and displays it. The word cloud is stored in a file as well.

        The function takes two lists of words, `self.clean_words` and `self.top_comm`, and creates a string by
        concatenating all the words. Then, it defines a set of stopwords from the `self.stop_words` list and a
        `WordCloud` object with the defined stopwords. The function generates a word cloud from the concatenated
        words string and stores it in a file with the name specified by `self.input_file` + '.png'. Finally,
        the function displays the generated word cloud using the `matplotlib` library.
        """
        all_words_string = ' '.join(self.clean_words + self.top_comm)
        all_stopwords_string = set(' '.join(self.stop_words))

        # defining the wordcloud parameters
        wc = WordCloud(background_color="white",
                       max_words=2000,
                       stopwords=all_stopwords_string)

        # generate words cloud
        wc.generate(all_words_string)

        # store to file
        _file = 'Outputs/{}{}'.format(self.input_filename, '.png')
        wc.to_file(filename=_file)

        # show the cloud
        plt.imshow(wc)
        plt.axis('off')
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Topics extraction using LDA (Latent Dirichlet Allocation)
    def topic_modeling(self):
        """
        This function performs topic modeling on a list of tokenized words (tokens).

        It creates a dictionary from the tokenized words using the Dictionary class from the gensim library, and
        converts the dictionary into a bag-of-words corpus representation using the doc2bow method. Then, it trains a
        Latent Dirichlet Allocation (LDA) model using the LdaModel class from the gensim library on the corpus
        representation. Finally, it displays the topics generated by the LDA model.

        If the visualize attribute of the current object is True, the function also creates a visual representation of
        the topics using the prepare method from the pyLDAvis library and the display method from pyLDAvis.display.
        """
        tokens = [x.split() for x in self.clean_words]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(text) for text in tokens]

        lda = LdaModel(corpus=corpus,
                       num_topics=self.num_topics,
                       id2word=dictionary)

        # print('\n the following are the top', num_topics, 'topics:')

        for i in range(0, len(lda.show_topics(self.num_topics))):
            print(lda.show_topics(self.num_topics)[i][1], '\n')

        if self.visualize:
            lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

            return pyLDAvis.display(lda_display)

    # ------------------------------------------------------------------------------------------------------------------
    # creates a matrix of co-occurrence
    def co_occurrence(self):
        """
        Generates a co-occurrence matrix from a list of text data.

        This function takes in a list of text data and a set of stop words. It performs preprocessing on the text by
        converting it to lowercase, splitting it into individual words, and filtering out words that are in the stop
        words set or have a length less than the minimum word length. The preprocessed words are then used to build a
        co-occurrence matrix where the value at row i and column j represents the number of times word i and word j
        co-occur within a specified word window size.

        The resulting co-occurrence matrix is returned as a pandas DataFrame.
        """
        d = defaultdict(int)
        vocab = set()
        # print (sentences)
        for text in self.txt_words:

            # preprocessing
            text = text.lower().split()
            self.get_unique_words(text, self.stop_words, self.min_word_len)
            for i in range(len(self.clean_words)):
                token = self.clean_words[i]
                vocab.add(token)  # add to vocab
                next_token = self.clean_words[i + 1: i + 1 + self.word_window]
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

    # ------------------------------------------------------------------------------------------------------------------
    # Sentiment Analysis
    @staticmethod
    def sentiment_analysis(text):
        """
        The method sentiment_analysis takes as input a string text, and it performs sentiment analysis on the input
        text. The method uses the SentimentIntensityAnalyzer from the nltk library to compute sentiment scores for
        the input text. The sentiment scores are represented as a dictionary, with the 'compound' key indicating the
        overall sentiment score, ranging from -1 (most negative) to 1 (most positive).

        Based on the value of the 'compound' key in the sentiment scores, the method returns one of three strings:
        'Positive' if the compound score is greater than or equal to 0.05, 'Negative' if the score is less than or equal
        to -0.05, and 'Neutral' if the score is between -0.05 and 0.05.
        """
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiment = sentiment_analyzer.polarity_scores(text)
        if sentiment['compound'] >= 0.05:
            return 'Positive'
        elif sentiment['compound'] <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    # ------------------------------------------------------------------------------------------------------------------
    # Summarize Text
    @staticmethod
    def summarize_text(text, language='english', ratio=0.5):
        """
        The function summarize_text is a text summarization utility that summarizes the input text using the Latent
        Semantic Analysis (LSA) algorithm. The input text is first parsed into a document using the PlaintextParser
        from the sumy library, and a tokenizer for the specified language is applied. The LsaSummarizer class is then
        used to generate a summary of the document, with the length of the summary determined by the ratio parameter.
        The summary is returned as a string, with each sentence separated by a newline character.
        """
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, ratio)
        summarized_text = '\n'.join([str(sentence) for sentence in summary])
        return summarized_text


def read_input(file):
    """
    Reads input from a text file and returns a list of its lines.

    :param file: The file to be read.
    :return: A list of lines from the input file.
    """
    file_name = file if file.lower().endswith('.txt') else file + '.txt'

    try:
        with open(file=file_name, mode='r', encoding='utf8') as fr1:
            return [word.strip() for word in fr1 if word]
    except FileNotFoundError:
        print('Error reading the input data')
        return None


# MAIN PROGRAM
if __name__ == '__main__':
    try:
        WordNetwork(input_file='Robert_Liu',  # Provide the input text file here
                    stop_file='stopwords_en.txt',  # Provide the stopword file here
                    visualize=False)
    except Exception as ex1:
        print('Something went wrong during execution: {}'.format(ex1))
