import pickle
import os
import io
import json 
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Embedding, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from src.utils import create_folder
from src.models.base import BaseModel


class BiLSTM(BaseModel):
    _model_name = 'BiLSTMClf'

    def __init__(self, output_dim=None, input_length=None, run_time=None, 
                       X_train=None, y_train=None, save_path=None, 
                       epochs=None, batch_size=None, validation_data=None, 
                       validation_split=None, verbose=1, embedding_path=None, 
                       n_words=10000):
        super().__init__()
        self._output_dim = output_dim
        self._input_length = input_length
        self._run_time = run_time
        self._X_train = X_train
        self._y_train = y_train
        self._save_path = save_path
        self._epochs = epochs
        self._batch_size = batch_size
        self._validation_data = validation_data
        self._validation_split = validation_split
        self._verbose = verbose
        self._embedding_path = embedding_path
        self._n_words = n_words
        self._input_dim = ''
        self._weights = ''
        self._tokenizer = ''
        self._model = None
        self._history = None

    def _create_model(self):
        self._model = Sequential([
            Embedding(
                input_dim=self._input_dim, # vocab_size
                output_dim=self._output_dim,
                weights=[self._weights], # embedding_matrix
                input_length=self._input_length,
                name='embeddings')]) # max_len

        self._model.add(
            Bidirectional(LSTM(64, return_sequences=True)))
        self._model.add(
            GlobalMaxPooling1D())
        self._model.add(
            Dense(16, activation='relu'))
        self._model.add(
            Dropout(0.30))
        self._model.add(
            Dense(6, activation='sigmoid'))

        self._model.compile(
            loss='binary_crossentropy',
            optimizer='adam', 
            metrics=['accuracy']) # TODO: change to correct metrics once all ok

    def _preprocess_data(self):
        self._tokenizer = Tokenizer(num_words=self._n_words, oov_token='<oov>')
        self._tokenizer.fit_on_texts(self._X_train)
        
        self._input_length = max([len(row) for row in self._X_train]) if self._input_length is None or self._input_length == 'None' else int(self._input_length)
        self._X_train = self._tokenize_and_pad(self._X_train, self._tokenizer, self._input_length)
        self._input_dim = len(self._tokenizer.word_index) + 1

    def _tokenize_and_pad(self, data, tokenizer, maxlen, padding='post', truncating='post'):
        print('tokenizing data')
        data = tokenizer.texts_to_sequences(data)
        print('padding tokens')
        return pad_sequences(data, maxlen=maxlen, padding=padding, truncating=truncating)

    def _get_embeddings(self):
        embeddings_index = {}
        print('reading pre-trained embeddings')
        glove = open(self._embedding_path, 'r', encoding='utf-8')
        for line in tqdm(glove):
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        glove.close()
        print('Found %s word vectors.' % len(embeddings_index))
        
        # creating embedding matrix for words dataset
        print('creating embedding matrix')
        self._weights= np.zeros((len(self._tokenizer.word_index)+1, self._output_dim))
        for word, index in tqdm(self._tokenizer.word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self._weights[index] = embedding_vector

    def get_summary(self):
        return self._model.summary()

    def train(self):

        # preprocess data
        print ('preprocessing data')
        self._preprocess_data()

        # get embeddings
        print('getting embeddings weights')
        self._get_embeddings()

        # create model architecture
        self._create_model()
        summary = self.get_summary()
        print('mode summary:', summary)

        self._save_path = os.path.join(self._save_path, 'checkpoints', f'{self._model_name}_{self._run_time}')
        create_folder(self._save_path)

        cp_callback = ModelCheckpoint(
            filepath=self._save_path,
            save_weights_only=False,
            # verbose=verbose,
            save_best_only=True,
            monitor='val_loss',
            mode='min')

        self._history = self._model.fit(
                                    self._X_train,
                                    self._y_train,
                                    epochs=self._epochs,
                                    validation_data=self._validation_data,
                                    validation_split=self._validation_split,
                                    batch_size=self._batch_size,
                                    callbacks=[cp_callback])

    def predict(self, X, threshhold=0.5):
        pred = self._model.predict(X)
        return (pred > threshhold).astype(np.int)

    def evaluate(self, X, y):
        X = self._tokenize_and_pad(X, self._tokenizer, self._input_length)
        return self._model.evaluate(X, y)

    def load_model(self, path):
        self._model = load_model(path)

    def save_model(self, mlflow, path):
        path = os.path.join(path, 'saved_models', f'{self._model_name}_{self._run_time}')
        create_folder(path)
        self._save_tokenizer(path)
        self._save_model(mlflow, path)
        self._save_embeddings(path)

    def _save_embeddings(self, path):
        embeddings = {}
        model_embeddings = self._model.get_layer('embeddings').get_weights()[0]
        for word, index in self._tokenizer.word_index.items():
            embeddings[word] = model_embeddings[index]
        with open(os.path.join(path, f'embeddings.pkl'), 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_tokenizer(self, path):
        tokenizer_json = self._tokenizer.to_json()
        with io.open(os.path.join(path, f'tokenizer.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def _save_model(self, mlflow, path):
        mlflow.keras.save_model(self._model, os.path.join(path, f'model'))
        