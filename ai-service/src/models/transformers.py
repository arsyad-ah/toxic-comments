import os
import torch
from src.pipelines import ToxicDataset
from src.models.base import BaseModel
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments


class BertSeqClf(BaseModel):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['DISABLE_MLFLOW_INTEGRATION'] = 'true'
    _MODEL_NAME = 'BertSeqClf'
    _MAX_LENGTH = 512
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    _BACKEND = ["bert-base-uncased"]
    print(f'device: {_DEVICE}')

    def __init__(self, backend, train_data=None, validation_data=None, train_config=None, mlflow=None):
        if backend not in self._BACKEND:
            raise ValueError(f'Unknown model selected. Please check value: {backend}')
        self._tokenizer, self._model = self.load_model(backend)
        self._model.config.problem_type = 'multi_label_classification'

        self._train_data, self._validation_data = self._prepare_data(
            train_data, validation_data)
        self._train_config = train_config
        self._train_config['output_dir'] = os.path.join(
            self._train_config['model_save_path'],
            self._MODEL_NAME,
            self._train_config['run_time']
        )
        self._trainer = None
        self._mlflow = mlflow

    def _prepare_data(self, train_data, validation_data):
        print('preparing data')
        train_td = ToxicDataset(train_data, self._tokenizer, self._MAX_LENGTH)
        print('train done')
        validation_td = ToxicDataset(validation_data, self._tokenizer, self._MAX_LENGTH)
        print('val done')
        return train_td, validation_td

    def train(self):
        self._training_args = TrainingArguments(
            output_dir=self._train_config['output_dir'],
            overwrite_output_dir=self._train_config['overwrite_output_dir'],
            num_train_epochs=self._train_config['num_train_epochs'],
            gradient_accumulation_steps=self._train_config['gradient_accumulation_steps'],
            per_device_train_batch_size=self._train_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self._train_config['per_device_eval_batch_size'],
            weight_decay=self._train_config['weight_decay'],
            load_best_model_at_end=self._train_config['load_best_model_at_end'],
            logging_steps=self._train_config['logging_steps'],
            evaluation_strategy=self._train_config['evaluation_strategy'],
            save_strategy=self._train_config['save_strategy'],
            logging_strategy=self._train_config['logging_strategy'],
            fp16=self._train_config['fp16'],
            fp16_opt_level=self._train_config['fp16_opt_level'],
            run_name=self._train_config['run_name']
        )

        print('training model')
        self._trainer = Trainer(
            model=self._model,
            tokenizer=self._tokenizer,
            args=self._training_args,
            train_dataset=self._train_data,
            eval_dataset=self._validation_data)
        
        self._trainer.train()
        return self._extract_train_history()
    
    def _extract_train_history(self):
        res = {}
        metrics_to_extract = ['loss', 'eval_loss', 'learning_rate', 'eval_runtime', ]
        for train_iter in self._trainer.state.log_history:
            for metric, value in train_iter.items():
                if metric in metrics_to_extract:
                    if metric in res:
                        res[metric].append(value)
                    else:
                        res[metric] = [value]
        return res

    def evaluate(self, eval_data):
        return self._trainer.evaluate(eval_data)

    def save_model(self):
        components = {
            "model": self._trainer.model,
            "tokenizer": self._trainer.tokenizer,
                    }
        self._mlflow.transformers.log_model(components,
                                      artifact_path='model',
                                      registered_model_name=self._MODEL_NAME,
                                      )

    def load_model(self, path):
        print('loading model')
        return BertTokenizerFast.from_pretrained(path, do_lower_case=True), \
            BertForSequenceClassification.from_pretrained(
                path,
                num_labels=len(ToxicDataset.get_labels())).to(self._DEVICE)

    def infer(self, model_path, model_selection, text):
        saved_model_path = os.path.join(model_path, 'saved_models', model_selection)
        tokenizer, model = self.load_model(saved_model_path)
        enc_text = tokenizer.encode(text, return_tensors='pt').to(self._BACKEND)
        pred = model(enc_text)
        return torch.argmax(pred.logits)
