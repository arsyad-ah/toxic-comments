import os
import torch
import nltk
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from src.utils import create_folder


class DataPipeline:
    def __init__(self, config, run_time):
        self._data = ''
        self._config = config
        self._run_time = run_time
        
    def read_data(self, file_path):
        self._data = self._read_data(
            file_path=self._config.get('PATHS', file_path), 
            source_type=os.path.splitext(self._config.get('PATHS', file_path))[1].strip('.')
        )
        print('Raw data shape:', self._data.shape)

    def _read_data(self, file_path, source_type):
        source_type = source_type.lower()
        if source_type == 'db':
            # TODO implement DB. use docker postgres
            pass
        elif source_type == 'csv':
            if self._config.get('DEFAULT', 'is_sample') == 'True':
                return pd.read_csv(file_path).head(50) # TODO remove once ok
            return pd.read_csv(file_path)
        else:
            raise Exception('Unknown file format. Please check')

    def clean_data(self):
        if 'id' in self._data.columns:
            self._data.drop('id', axis=1, inplace=True)
        self._data['toxic_cum_sum'] = self._data.iloc[:, 1:].sum(axis=1)
        self._data['is_negative'] = self._data['toxic_cum_sum']\
            .apply(lambda x: 1 if x >= 1 else 0)

        self._data['sentence_count'] = self._data["comment_text"]\
            .apply(self._get_sentence_count)

        self._data['count_word'] = self._data["comment_text"]\
            .apply(lambda x: len(str(x).split()))

        self._data['count_unique_word'] = self._data["comment_text"]\
            .apply(lambda x: len(set(str(x).split())))

        self._data['comment_text_clean'] = self._data['comment_text']\
            .apply(lambda x: x.replace('\r', ' ').replace('\n', ' ')\
                .replace('\t', ' ').replace("'", "").replace(",", "")\
                    .encode('ascii', errors='ignore').decode())# if type(x) is str else x)
        
        self._data['char_count'] = self._data['comment_text_clean'].apply(self._get_char_count)
        self._data['unq_char_count'] = self._data['comment_text_clean'].apply(self._get_unq_char_count)
        self._data['tok_clean_comments'] = self._get_tok_clean_comments(add_stopwords=None)


    def _get_sentence_count(self, x):
        return len(sent_tokenize(x))

    def _get_tok_clean_comments(self, add_stopwords=None):
        print ('add_stopwords:', add_stopwords)
        tok_clean_comments = []
        stop_words = set(stopwords.words('english'))
        if add_stopwords is None:
            pass
        elif type(add_stopwords) == list:
            for i in add_stopwords:
                stop_words.add(i)
        else:
            raise ValueError('Unknown input. Please check.')
        lemmatizer = WordNetLemmatizer()
        print('lemmatizing tokens')
        for i in tqdm(range(self._data.shape[0])):
            comments = word_tokenize(self._data.loc[i, 'comment_text_clean'])
            tok_clean_comments.append([lemmatizer.lemmatize(w.lower()) for w in comments \
                                    if w.isalpha() and w.lower() not in stop_words])
        return tok_clean_comments

    def _get_word_count(self, comment):
        word_count = {}
        for word in comment:
            if word in word_count.keys():
                word_count[word] += 1
            else:
                word_count[word] = 1
        return word_count

    def _get_char_count(self, x):
        return len(x.replace(' ', ''))

    def _get_unq_char_count(self, x):
        return len(set(x.replace(' ', '')))

    def prepare(self):
        X_col = 'comment_text'
        y_col = self._data.drop(X_col, axis=1).columns

        train, test = self._split_data(
            X_col, 
            y_col, 
            float(self._config.get('DEFAULT', 'train_size')), 
            int(self._config.get('DEFAULT', 'random_seed')))
        
        # meta_data = {
        #     f'train_{self._run_time}.csv': train, 
        #     f'test_{self._run_time}.csv': test
        # }

        meta_data = {
            f'train.csv': train, 
            f'test.csv': test
        }

        for k,v in meta_data.items():
            self._save(v, self._config.get('PATHS', 'interim_data_path'), k)

    def _save(self, data, path, file_name):
        create_folder(path)
        print(f'saving interim dataset: {file_name}')
        data.to_csv(os.path.join(path, file_name), index=False)

    def _split_data(self, X_col, y_col, train_size, random_seed, stratify=False):
        # TODO: chec logic for below and reason for code
        # if type(X_col) != list and type(y_col) != list:
        #     raise Exception(f'Expecting a list of column names, \
        #         received X_col: {type(X_col)} and y_col: {type(y_col)}. Please check')
        
        # X_train, X_test, y_train, y_test = train_test_split(
        #     self._data[X_col],
        #     self._data[y_col],
        #     train_size=train_size,
        #     random_state=random_seed,
        #     stratify=self._data[y_col])

        # return X_train, X_test, y_train, y_test
        stratify_by = self._data[y_col] if stratify else None
        
        train, test = train_test_split(
            self._data,
            train_size=train_size,
            random_state=random_seed,
            stratify=stratify_by)

        return train, test


class ToxicDataset(Dataset):
    _x_col = 'comment_text_clean'
    _y_col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    def __init__(self, data, tokenizer, max_len):
        self._data = data
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._texts = []
        self._labels = []
        
        for idx in range(len(self._data)):
            self._texts.append(self._data.loc[idx, self._x_col])
            self._labels.append(self._data.loc[idx, self._y_col].tolist())
        
        
    def __len__(self):
        return len(self._data)
    
    def _prepare_data(self, i):
        return self._tokenizer.encode_plus(self._texts[i],
                                add_special_tokens=True,
                              padding=True,
                              truncation='longest_first',
                              max_length=self._max_len,
                               return_attention_mask=True,
                               return_token_type_ids=True,
                              return_tensors='pt')
    
    def __getitem__(self, item):
        tok_text = self._prepare_data(item)
        return {
            'input_ids': tok_text['input_ids'].flatten(),
            'token_type_ids': tok_text['token_type_ids'].flatten(),
            'attention_mask': tok_text['attention_mask'].flatten(),
            'label': torch.tensor(self._labels[item], dtype=torch.float32)
        }