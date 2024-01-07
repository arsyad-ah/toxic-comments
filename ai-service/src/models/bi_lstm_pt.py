import os
import io
import json 
import mlflow
import pickle
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from src.utils import create_folder
from src.models.base import BaseModel

# TODO: implement pytorch version of model
class BiLSTMClfPT(nn.ModuleList, BaseModel):
    _MODEL_NAME = 'BiLSTMClfPT'
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, train_data, validation_data, train_config):
        super().__init__()
        self._X_train = train_data.iloc[:, 0]
        self._y_train = train_data.iloc[:, 1:7].to_numpy()
        self._X_val = validation_data.iloc[:, 0]
        self._y_val = validation_data.iloc[:, 1:7] .to_numpy()
        self._train_config = train_config
        self._input_dim = None
        self._weights = None
        self._tokenizer = None
        self._model = None
        self._history = None
        self._extract_train_config()

    def _extract_train_config(self):
        self._dropout = self._train_config['dropout']
        self._epochs = self._train_config['epochs']
        self._run_time = self._train_config['run_time']
        self._n_layers = self._train_config['n_layers']
        self._batch_size = self._train_config['batch_size']
        self._output_dim = self._train_config['output_dim']
        self._hidden_dim = self._train_config['hidden_dim']

        self._input_length = self._train_config['input_length']
        self._verbose = self._train_config['verbose']
        self._embedding_path = self._train_config['embedding_path']
        self._n_words = self._train_config['n_words']

        self._model_save_path = os.path.join(
            self._train_config['model_save_path'],
            self._MODEL_NAME,
            self._run_time
        )

    def _create_model(self):

        # embedding layer
        self._embedding = nn.Embedding(self._n_words, self._embedding_dim)
        
        #lstm layer
        self._lstm = nn.LSTM(self._embedding_dim,
                           self._hidden_dim,
                           num_layers=self._n_layers,
                           bidirectional=self._bidirectional,
                           dropout=self._dropout,
                           batch_first=True)
        # dense layer
        self._fc = nn.Linear(self._hidden_dim * 2, self._output_dim)

        #activation function
        self._act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        
        embedded = self._embedding(text)
      
        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        
        _, (hidden, _) = self.lstm(packed_embedded)

        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
                
        #hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.act(dense_outputs)
        
        return outputs

    def _preprocess_data(self):
        pass

    def _tokenize_and_pad(self, data, tokenizer, maxlen, padding='post', truncating='post'):
        pass

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
        self._weights = np.zeros((len(self._tokenizer.word_index)+1, self._output_dim))
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

        create_folder(self._model_save_path)

        cp_callback = ModelCheckpoint(
            filepath=os.path.join(self._model_save_path, 'checkpoint'),
            save_weights_only=False,
            # verbose=verbose,
            save_best_only=True,
            monitor='val_loss',
            mode='min')

        self._history = self._model.fit(
                                    self._train,
                                    epochs=self._epochs,
                                    validation_data=self._val,
                                    batch_size=self._batch_size,
                                    callbacks=[cp_callback])
        return self._history.history
        
        
    def predict(self, X, threshhold=0.5):
        pred = self._model.predict(X)
        return (pred > threshhold).astype(np.int)

    def evaluate(self, eval_data):
        X, y = eval_data.iloc[:, 0], eval_data.iloc[:, 1:7].to_numpy()
        X = self._tokenize_and_pad(X, self._tokenizer, self._input_length)
        return self._model.evaluate(X, y)

    def load_model(self, path):
        self._model = load_model(path)

    def save_model(self):
        path = os.path.join(self._model_save_path, 'model')
        create_folder(path)
        print('saving tokenizer')
        self._save_model(path)
        print('saving embeddings')
        self._save_tokenizer(path)
        print('saving model')
        self._save_embeddings(path)

    def _save_embeddings(self, path):
        embeddings = {}
        model_embeddings = self._model.get_layer('embeddings').get_weights()[0]
        for word, index in self._tokenizer.word_index.items():
            embeddings[word] = model_embeddings[index]
        with open(os.path.join(path, 'embeddings.pkl'), 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_tokenizer(self, path):
        tokenizer_json = self._tokenizer.to_json()
        with io.open(os.path.join(path, 'tokenizer.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def _save_model(self, path):
        mlflow.tensorflow.save_model(self._model, path)
        mlflow.tensorflow.log_model(
            sk_model=self._model,
            artifact_path=path,
            registered_model_name=self._MODEL_NAME,
    )
        