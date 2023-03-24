import os
import pickle
import json
from src.datapipeline import DataPipeline
from src.models import BiLSTMClf, BertSeqClf
from keras_preprocessing.text import tokenizer_from_json

MAXLEN=100

def make_inference(config, run_time):

    # load inference data
    dpl = DataPipeline(config, run_time)
    dpl.read_data('inference_data')
    inference_data = dpl._data
    print('inference_data shape: ', inference_data.shape)

    tokenizer = _load_tokenizer(config)
    print('tokenizer loaded')


    # load model
    model = BiLSTM()

    encoded_inference_data = model._tokenize_and_pad(inference_data.comment_text, tokenizer, maxlen=MAXLEN)
    print('data preprocessed for inference')

    model.load_model(
        os.path.join(
            config.get('PATHS', 'model_path'),
            'saved_models',
            config.get('INFERENCE', 'inference_model'),
            'model'
            ))

    print('model loaded')

    prediction = model.predict(encoded_inference_data[0])
    print (prediction)


def _load_tokenizer(config):
    with open(os.path.join(
        config.get('PATHS', 'model_path'),
        'saved_models',
        config.get('INFERENCE', 'inference_model'),
        'tokenizer.json'),
        'rb') as f:
        tok = json.load(f)
    return tokenizer_from_json(tok)