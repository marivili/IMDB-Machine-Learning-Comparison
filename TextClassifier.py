from collections import Counter
import numpy as np
from nltk.corpus import movie_reviews  
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import fetch_openml
import tensorflow_datasets as tfds, tensorflow as tf
import os, sys, re, nltk 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

tf.get_logger().setLevel('ERROR') 

class TextClassifier:
    def __init__(self, vocab_size=10000, test_size=0.2, dev_size=0.125, n_common=100, k_rare=100, m_features=5000):
        self.vocab_size = vocab_size
        self.test_size = test_size
        self.dev_size = dev_size
        self.n_common = n_common
        self.k_rare = k_rare
        self.m_features = m_features

        print("Loading dataset and building vocabulary")
        self.load_dataset()
        self.build_vocab()

        print("Selecting top features")
        self.select_top_m_features()
        self.process_data()

    def load_dataset(self):
        dataset, info = tfds.load("imdb_reviews", split=["train", "test"], as_supervised=True, with_info=True)
    
        self.texts = []
        self.labels = []
    
        for text, label in tfds.as_numpy(dataset[0]):  
            self.texts.append(text.decode("utf-8"))
            self.labels.append(int(label))
    
        for text, label in tfds.as_numpy(dataset[1]): 
            self.texts.append(text.decode("utf-8"))
            self.labels.append(int(label))

  
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def build_vocab(self):
        word_counter = Counter()

        for text in self.texts:
            words = set(self.clean_text(text).split())  
            word_counter.update(words)

        most_common_words = set([word for word, _ in word_counter.most_common(self.n_common)])
        least_common_words = set([word for word, _ in word_counter.most_common()[:-self.k_rare-1:-1]])

        filtered_words = [(word, count) for word, count in word_counter.items() if word not in most_common_words and word not in least_common_words]

        self.vocab = {word: i for i, (word, _) in enumerate(sorted(filtered_words, key=lambda x: x[1], reverse=True)[:self.vocab_size])}

    def select_top_m_features(self):
        print("Computing Information Gain...")

        num_preselected_features = min(3000, len(self.vocab)) 
        preselected_words = list(self.vocab.keys())[:num_preselected_features]
        self.vocab = {word: i for i, word in enumerate(preselected_words)}

        X_temp = np.array([self.text_to_vector(text) for text in self.texts], dtype=np.int8)
        y_temp = np.array(self.labels, dtype=np.int8)

        info_gains = mutual_info_classif(X_temp, y_temp, discrete_features=True)

        top_m_indices = np.argsort(info_gains)[-self.m_features:]
        self.vocab = {word: i for i, word in enumerate(np.array(preselected_words)[top_m_indices])}


    def text_to_vector(self, text):
    
        vector = np.zeros(len(self.vocab), dtype=int) 
        words = set(self.clean_text(text).split())

        for word in words:
            if word in self.vocab:
                idx = self.vocab[word]
                if idx < len(vector):  
                    vector[idx] = 1

        return vector

    def process_data(self):
        X = np.array([self.text_to_vector(text) for text in self.texts])
        y = np.array(self.labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

    
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=self.dev_size/(1-self.test_size), random_state=42)

        self.X_train, self.X_dev, self.X_test = X_train, X_dev, X_test
        self.y_train, self.y_dev, self.y_test = y_train, y_dev, y_test

        print(f"Dataset split: Train: {len(y_train)}, Dev: {len(y_dev)}, Test: {len(y_test)}")

    def get_data(self):
        return self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, self.y_test
