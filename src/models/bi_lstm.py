import pickle
import os
import io
import json 
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Embedding, GlobalMaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from src.utils import create_folder
from src.models.base import BaseModel


class BiLSTM(BaseModel):
    _model_name = 'BiLSTMClf'

    def __init__(self, input_dim=None, output_dim=None, weights=None, input_length=None, run_time=None, tokenizer=None):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._weights = weights
        self._input_length = input_length
        self._run_time = run_time
        self._tokenizer = tokenizer
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
            metrics=['accuracy', 'crossentropy']) # TODO: change to correct metrics once all ok

    def get_summary(self):
        return self._model.summary()

    def train(self,
              X_train,
              y_train,
              save_path,
              epochs=1,
              batch_size=16,
              validation_data=None,
              validation_split=None,
              verbose=1):

        self._create_model()
        summary = self.get_summary()
        print('mode summary:', summary)

        save_path = os.path.join(save_path, 'checkpoints', f'{self._model_name}_{self._run_time}')
        create_folder(save_path)

        cp_callback = ModelCheckpoint(
            filepath=save_path,
            save_weights_only=False,
            # verbose=verbose,
            save_best_only=True,
            monitor='val_loss',
            mode='min')

        self._history = self._model.fit(
                                    X_train,
                                    y_train,
                                    epochs=epochs,
                                    validation_data=validation_data,
                                    validation_split=validation_split,
                                    batch_size=batch_size,
                                    callbacks=[cp_callback])

    def predict(self, X, threshhold=0.5):
        pred = self._model.predict(X)
        return (pred > threshhold).astype(np.int)

    def evaluate(self, X, y):
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
        