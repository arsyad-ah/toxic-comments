import os
import pandas as pd
import numpy as np
from src.utils import cfg_to_dict
from src.models.bi_lstm import BiLSTM
from src.datapipeline import DataPipeline
import mlflow
    

def train(config, run_time):
    # mlflow_path = os.path.join(os.getcwd(), 'mlflow')
    # uri = f'file://{mlflow_path}'
    # mlflow.set_tracking_uri(uri)

    with mlflow.start_run():
        print('Starting model training')
        dpl = DataPipeline(config, run_time)
        
        # TODO: convert to DB
        dpl.read_data('interim_train_data')
        train_data = dpl._data
        print('train:', train_data.shape)
        
        # TODO: convert to DB
        dpl.read_data('interim_test_data')
        test_data = dpl._data
        print(test_data.shape)

        print('Getting model selection')
        model_selection = int(config.get('DEFAULT', 'model_selection'))

        # splitting data
        print('splitting X, y data')
        X_train = train_data['comment_text']
        y_train = train_data.iloc[:, 1:7]
        X_test = test_data['comment_text']
        y_test = test_data.iloc[:, 1:7]

        # get model params
        print('getting model params')
        model_params = _get_model_params(config, model_selection, no_defaults=True)
        model_params['run_time'] = run_time
        print ('model_params', model_params)

        # init model
        print('model init')
        model = BiLSTM(
            X_train=X_train,
            y_train=y_train,
            validation_data=None,
            **model_params)
        
        # train model
        print('training model')
        model.train()
        
        # evaluate TODO: create another split from train
        print('evaluating model')
        evaluation = model.evaluate(X_test, y_test)
        print('evaluation', evaluation)

        # save
        print('saving model')
        model.save_model(mlflow, model_params['save_path'])

                
        print('logging params')
        mlflow.log_params(model_params)
        print (model._history.history)
        print('logging metrics')
        mlflow.log_metrics({'loss': evaluation[0],
                            'accuracy': evaluation[1]})

        mlflow.end_run()
    return

def _get_model_params(config, model_selection, no_defaults=False):
    if model_selection == 1:
        section = 'LSTM_MODEL'
    return cfg_to_dict(config, section, no_defaults=no_defaults)
