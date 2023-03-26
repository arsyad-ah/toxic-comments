import os
import mlflow
import pandas as pd
import numpy as np
from src.utils import cfg_to_dict, read_training_params, create_training_params
from src.models import BiLSTMClf, BertSeqClf
from src.datapipeline import DataPipeline
    

MLFLOW_TRACKING_URI = f'http://{os.environ["MLFLOW_NAME"]}:{os.environ["MLFLOW_PORT"]}'
EXPERIMENT_NAME = 'toxic-comments-exp'


def train(config, run_time):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
        print('Existing experiment')
    except mlflow.exceptions.RestException as err:
        print(err)
        print('Creating experiment')
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print('Experiment created')

    with mlflow.start_run(run_name=run_time, experiment_id=experiment_id):
        try:
            print('Starting model training')
            dpl = DataPipeline(config, run_time)

            # TODO: convert to DB
            dpl.read_data('train_data')
            train_data = dpl._data
            train_data = train_data.sample(100)
            train_data.reset_index(inplace=True, drop=True)
            print('train shape: ', train_data.shape)

            # TODO: convert to DB
            dpl.read_data('test_data')
            val_data = dpl._data
            val_data = val_data.sample(5)
            val_data.reset_index(inplace=True, drop=True)
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
            train_history = model.train()

            # save
            print('saving model')
            model.save_model()


            print('logging params')
            mlflow.log_params(train_params)
            print('logging metrics')
            for metric in train_history.keys():
                for ep in range(len(train_history[metric])):
                    mlflow.log_metric(key=metric, value=train_history[metric][ep], step=ep)

        except Exception as err:
            print(f'Error: {err}')
        finally:
            mlflow.end_run()