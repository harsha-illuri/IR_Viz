import pickle

import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Embedding, Dropout, MaxPooling1D, LSTM
from keras.utils import to_categorical
import matplotlib.pyplot as plt
pd.set_option('display.expand_frame_repr', False)
from keras.models import load_model

class Classifier_LSTM():
    def __init__(self, dataFramePath):
        self.df = pd.read_hdf(dataFramePath, key='key', start = 0, stop=10000)
        self.class_map = {'Information': 0, 'Communication': 1, 'Resources': 2, 'Procedural&Team': 3, 'Control': 4,
                     'Cognition': 5, 'Management': 6, 'Performance': 7, 'Org-Coordination': 8, 'Coordination': 9,
                     'Team & Organization': 10, 'Planning': 11}
        self.max_length = 100
        self.vocabulary_size = 0
        self.lstm_size = 20
        self.class_rev_map = {v: k for k, v in self.class_map.items()}
        pass

    def getClassName(self, row):
        return self.class_rev_map[row]

    def getTokenizer(self, save = False):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.df['text'])
        self.vocabulary_size = len(tokenizer.word_index) + 1
        if save:
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return tokenizer

    def getPaddedSent(self, tokenizer):
        encoded_sent = tokenizer.texts_to_sequences(self.df['text'])
        padded_sent = pad_sequences(encoded_sent, self.max_length, padding='post')
        return padded_sent

    def getGloveEmbeddings(self, tokenizer):
        embeddings_index = dict()
        f = open('glove.6B.100d.txt', encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((self.vocabulary_size, 100))
        for word, index in tokenizer.word_index.items():
            if index > self.vocabulary_size - 1:
                break
            else:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
        return embedding_matrix

    def getModel(self, embedding_matrix):
        model = Sequential()
        model.add(Embedding(self.vocabulary_size, 100, input_length=self.max_length, weights=[embedding_matrix], trainable=True))
        model.add(Dropout(0.2))
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(self.lstm_size))
        model.add(Dense(12, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def assign_category(self,row):
        return self.class_map[row]

    def cleanText(self, line):
        # Converting to lower
        line = line.lower()

        # Removing alphanumerics
        tokens = [word for word in line.split() if word.isalpha()]

        # Removing Punctuations
        translator = str.maketrans("", "", string.punctuation)
        tokens = [word.translate(translator) for word in tokens]

        # Removing stop_words
        # stop_words = set(stopwords.words('english'))
        # tokens = [word for word in tokens if not word in stop_words]

        # Removing short_words
        tokens = [word for word in tokens if len(word) > 1]
        return tokens

    def trainOnData(self, save = False, graphs = False):
        tokenizer = self.getTokenizer()
        embedding_matrix = self.getGloveEmbeddings(tokenizer)
        padded_sent = self.getPaddedSent(tokenizer)

        Y = self.df['class'].apply(self.assign_category)
        Y = to_categorical(Y, 12)
        model = self.getModel(embedding_matrix)

        history = model.fit(padded_sent, Y, batch_size=64, epochs=10)

        if save:
            model.save('lstm.h5')

        if graphs:
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            
    def predict(self):
        tokenizer = self.getTokenizer()
        padded_sent = self.getPaddedSent(tokenizer)
        model = load_model('lstm.h5')
        res = model.predict(padded_sent)
        prob = [round(max(r), 4) for r in res]
        labels = np.argmax(res, axis=-1)
        self.df['prob'] = prob
        self.df['predict'] = labels
        self.df['predict'] = self.df['predict'].apply(self.getClassName)
        self.df = self.df.loc[self.df['prob'] > 0.9]
        # self.df.to_csv('testt.csv')
        df_raw = pd.read_hdf('raw_data.h5',key='raw')
        df_raw = pd.merge(df_raw, self.df, how='left', on=['file', 'line', 'start_pos', 'end_pos'] )
        df_raw.drop(['text_y'], axis = 1, inplace=True)
        df_raw.to_csv('test.csv', index=False)
        return df_raw







