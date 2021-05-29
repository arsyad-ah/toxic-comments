import os
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from src.utils import _cfg_to_dict
from src.models.bi_lstm import BiLSTM
from src.datapipeline import DataPipeline
import mlflow
    

def train(config, run_time):
    # mlflow_path = os.path.join(os.getcwd(), 'mlflow')
    # uri = f'file://{mlflow_path}'
    # mlflow.set_tracking_uri(uri)

    with mlflow.start_run():
        dpl = DataPipeline(config, run_time)
        # TODO: convert to DB
        dpl.read_data('interim_train_data')
        train_data = dpl._data

        # TODO: convert to DB
        dpl.read_data('interim_test_data')
        test_data = dpl._data
        print(test_data.shape)

        model_selection = int(config.get('DEFAULT', 'model_selection'))

        # get model params
        model_params = _get_model_params(config, model_selection)
        model_params['run_time'] = run_time

        # preprocess
        X_train = train_data['comment_text']
        y_train = train_data.iloc[:, 1:7]
        X_test = test_data['comment_text']
        y_test = test_data.iloc[:, 1:7]
        print('splitting train, test data')
        
        tokenizer, training_padded, validation_padded, maxlen = _preprocess_data(X_train, X_test,  maxlen=config.get('LSTM_MODEL', 'maxlen'))
        print ('preprocessing data')

        model_params['input_dim'] = len(tokenizer.word_index) + 1
        model_params['input_length'] = maxlen
        embedding_weights = _get_embeddings(tokenizer,
                                            model_params['embedding_path'],
                                            model_params['output_dim'])
        print('getting embeddings weights')
        mlflow.tensorflow.autolog()
        # init model
        model = BiLSTM(
            weights=embedding_weights,
            input_dim=model_params['input_dim'], 
            output_dim=model_params['output_dim'], 
            input_length=model_params['input_length'],
            run_time=model_params['run_time'],
            tokenizer=tokenizer)
        print('model init')
        
        # train model
        model.train(X_train=training_padded,
                y_train=y_train,
                save_path=model_params['save_path'],
                epochs=model_params['epochs'],
                batch_size=model_params['batch_size'],
                validation_split=0.2,
                verbose=model_params['verbose'])
        print('model trained')
        
        # evaluate TODO: create another split from train
        evaluation = model.evaluate(validation_padded, y_test)
        print('evaluation', evaluation)

        # save
        model.save_model(mlflow, model_params['save_path'])
        print('model saved')

        #test predict
        predictions = model.predict(validation_padded)

        print ('model_params', model_params)
        print('logging params')
        mlflow.log_params(model_params)
        print('logging metrics')
        print (model._history.history)
        mlflow.log_metrics({'loss': evaluation[0],
                            'accuracy': evaluation[1]})

        mlflow.end_run()
    return


def _get_model_params(config, model_selection):
    if model_selection == 1:
        section = 'LSTM_MODEL'
    return _cfg_to_dict(config, section)


def _preprocess_data(X_train, X_val, maxlen, n_words=100000):
    tokenizer = Tokenizer(num_words=n_words, oov_token='<oov>')
    tokenizer.fit_on_texts(X_train)
    
    maxlen = max([len(row) for row in X_train]) if maxlen is None or maxlen == 'None' else int(maxlen)

    training_padded = _tokenize_and_pad(X_train, tokenizer, maxlen)
    validation_padded = _tokenize_and_pad(X_val, tokenizer, maxlen)

    return tokenizer, training_padded, validation_padded, maxlen

def _tokenize_and_pad(data, tokenizer, maxlen, padding='post', truncating='post'):
    data = tokenizer.texts_to_sequences(data)
    return pad_sequences(data, maxlen=maxlen, padding=padding, truncating=truncating)


def _get_embeddings(tokenizer, embeddings_path, dim=200):
    embeddings_index = {}
    glove = open(embeddings_path,'r',encoding='utf-8')
    for line in tqdm(glove):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    glove.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # creating embedding matrix for words dataset
    embedding_matrix = np.zeros((len(tokenizer.word_index)+1, dim))
    for word, index in tqdm(tokenizer.word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix