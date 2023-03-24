import os
import mlflow
import pandas as pd
import numpy as np
from src.utils import cfg_to_dict, read_training_params, create_training_params
from src.models import BiLSTMClf, BertSeqClf
from src.datapipeline import DataPipeline
    

def train(config, run_time):
    # mlflow_path = os.path.join(os.getcwd(), 'mlflow')
    # uri = f'file://{mlflow_path}'
    # mlflow.set_tracking_uri(uri)

    with mlflow.start_run():
        print('Starting model training')
        dpl = DataPipeline(config, run_time)
        
        # TODO: convert to DB
        dpl.read_data('train_data')
        train_data = dpl._data
        print('train shape: ', train_data.shape)
        
        # TODO: convert to DB
        dpl.read_data('test_data')
        val_data = dpl._data
        print('test shape: ', val_data.shape)

        print('Getting model selection')
        model_selection = config.get('DEFAULT', 'model_selection')

        print('getting model params')
        train_config = read_training_params(
            config.get(
                'PATHS', 
                'train_config_path'
            )
        )

        # init model
        print('model init')
        train_params = create_training_params(
            config,
            train_config,
            model_selection,
            run_time
        )

        if model_selection == 'BiLSTMClf':
            model = BiLSTMClf(train_data, val_data, train_params)

        elif model_selection == 'BertSeqClf':
            model = BertSeqClf('bert-base-uncased', train_data, val_data, train_params)

        else:
            raise NotImplementedError
        
        # train model
        print('training model')
        model.train()
        
        # # save
        # print('saving model')
        # model.save_model(mlflow)

                
        # print('logging params')
        # mlflow.log_params(model_params)
        # print (model._history.history)
        # print('logging metrics')
        # mlflow.log_metrics({'loss': evaluation[0],
        #                     'accuracy': evaluation[1]})

        mlflow.end_run()
    return

