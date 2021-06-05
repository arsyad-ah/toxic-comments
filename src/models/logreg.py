import os
import joblib
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.models.base import BaseModel
from src.utils import create_folder

class LogReg(BaseModel):
    _model_name = 'LRClf'

    def __init__(self, run_time=None, X_train=None, y_train=None, 
                       save_path=None, embedding_method=None, validation_data=None, 
                       embedding_path=None, ):
            super().__init__()
            self._run_time = run_time
            self._X_train = X_train
            self._y_train = y_train
            self._save_path = save_path
            self._validation_data = validation_data
            self._embedding_method = embedding_method.lower()
            self._embedding_path = embedding_path
            self._model = None

    def _prepare_full_pipeline(self):
        if self._embedding_method == 'tfidf':
            vectorizer = TfidfVectorizer()
        elif self._embedding_method == 'cvec':
            vectorizer = CountVectorizer()
        self._model = Pipeline([('vectorizer', vectorizer), ('lr', LogisticRegression())])

    def train(self):
        self._model.fit(self._X_train, self._y_train)

    def evaluate(self, X, y):
        return self._model.score(X, y)


    def save_model(self, mlflow, path):
        path = os.path.join(path, 'saved_models', f'{self._model_name}_{self._run_time}')
        create_folder(path)
        self._save_model(mlflow, path)

    def _save_model(self, mlflow, path):
        mlflow.sklearn.save_model(self._model, os.path.join(path, f'model'))

    def load_model(self, path):
        self._model = joblib.load(path)

    def predict(self, X, threshhold=0.5):
        return self._model.predict_proba(X) > threshhold
