import os
import json
import pickle
import pandas as pd
from .data import DataPipeline
from src.models import BiLSTMClfTF, BertSeqClf
from keras_preprocessing.text import tokenizer_from_json


MAXLEN = 100

def make_inference(config, run_time):

    # load inference data
    inference_dpl = DataPipeline(config, run_time)
    inference_dpl.read_data('inference_data')
    print('inference_data shape: ', inference_dpl.data.shape)

    model_selection = config.get('INFERENCE', 'inference_model')
    run_id = config.get('INFERENCE', 'run_id')
    model_name = model_selection.split('_')[0]

    if model_name == 'BiLSTMClfTF':
        model = BiLSTMClfTF()
    elif model_name == 'BertSeqClf':
        model = BertSeqClf('bert-base-uncased')
    else:
        raise NotImplementedError

    model_path = config.get('PATHS', 'model_path')
    prediction = model.infer(model_path, model_selection, inference_dpl.data, MAXLEN)

    print (prediction)
