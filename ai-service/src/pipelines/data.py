import os
import torch
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from src.utils import create_folder, FileFormatError


class DataPipeline:
    def __init__(self, config, run_time):
        self.data = ''
        self._config = config
        self._run_time = run_time

    def read_data(self, file_path):
        self.data = self._read_data(
            file_path=self._config.get('PATHS', file_path),
            source_type=os.path.splitext(
                self._config.get('PATHS', file_path))[1].strip('.')
        )
        print('Raw data shape:', self.data.shape)

    def _read_data(self, file_path, source_type):
        source_type = source_type.lower()
        if source_type == 'db':
            # TODO implement DB. use docker postgres
            raise NotImplementedError('Source not implemented')
        if source_type == 'csv':
            if self._config.get('DEFAULT', 'is_sample') == 'True':
                return pd.read_csv(file_path)
            return pd.read_csv(file_path)
        raise FileFormatError('Unknown file format. Please check')

    def clean_data(self):
        if 'id' in self.data.columns:
            self.data.drop('id', axis=1, inplace=True)
        self.data['toxic_cum_sum'] = self.data.iloc[:, 1:].sum(axis=1)
        self.data['is_negative'] = self.data['toxic_cum_sum']\
            .apply(lambda x: 1 if x >= 1 else 0)

        self.data['sentence_count'] = self.data["comment_text"]\
            .apply(self._get_sentence_count)

        self.data['count_word'] = self.data["comment_text"]\
            .apply(lambda x: len(str(x).split()))

        self.data['count_unique_word'] = self.data["comment_text"]\
            .apply(lambda x: len(set(str(x).split())))

        self.data['comment_text_clean'] = self.data['comment_text']\
            .apply(lambda x: x.replace('\r', ' ').replace('\n', ' ')\
                .replace('\t', ' ').replace("'", "").replace(",", "")\
                    .encode('ascii', errors='ignore').decode())# if type(x) is str else x) # noqa: E501
        
        self.data['char_count'] = self.data['comment_text_clean'].apply(self._get_char_count) # noqa: E501
        self.data['unq_char_count'] = self.data['comment_text_clean'].apply(self._get_unq_char_count) # noqa: E501
        self.data['tok_clean_comments'] = self._get_tok_clean_comments(add_stopwords=None) # noqa: E501

    def _get_sentence_count(self, x):
        return len(sent_tokenize(x))

    def _get_tok_clean_comments(self, add_stopwords=None):
        print ('add_stopwords:', add_stopwords)
        tok_clean_comments = []
        stop_words = set(stopwords.words('english'))
        if add_stopwords is None:
            pass
        elif isinstance(add_stopwords, list):
            for i in add_stopwords:
                stop_words.add(i)
        else:
            raise ValueError('Unknown input. Please check.')
        lemmatizer = WordNetLemmatizer()
        print('lemmatizing tokens')
        for i in tqdm(range(self.data.shape[0])):
            comments = word_tokenize(self.data.loc[i, 'comment_text_clean'])
            tok_clean_comments.append([lemmatizer.lemmatize(w.lower()) for w in comments \
                                    if w.isalpha() and w.lower() not in stop_words])  # noqa: E501
        return tok_clean_comments

    def _get_word_count(self, comment):
        word_count = {}
        for word in comment:
            if word in word_count.keys():
                word_count[word] += 1
            else:
                word_count[word] = 1
        return word_count

    def _get_char_count(self, char):
        return len(char.replace(' ', ''))

    def _get_unq_char_count(self, x):
        return len(set(x.replace(' ', '')))

    def prepare(self):
        x_col = 'comment_text'
        y_col = self.data.drop(x_col, axis=1).columns

        train, test = self._split_data(
            y_col,
            float(self._config.get('DEFAULT', 'train_size')),
            int(self._config.get('DEFAULT', 'random_seed')))

        meta_data = {
            'train.csv': train, 
            'test.csv': test
        }

        train_file_path = self._config.get('PATHS', 'train_data_path')
        for file_name, data in meta_data.items():
            self._save(data, train_file_path, file_name)

    def _save(self, data, path, file_name):
        create_folder(path)
        print(f'saving interim dataset: {file_name}')
        data.to_csv(os.path.join(path, file_name), index=False)

    def _split_data(self, y_col, train_size, random_seed, stratify=False):
        # TODO: chec logic for below and reason for code
        # if type(X_col) != list and type(y_col) != list:
        #     raise Exception(f'Expecting a list of column names, \
        #         received X_col: {type(X_col)} and y_col: {type(y_col)}. Please check')

        # X_train, X_test, y_train, y_test = train_test_split(
        #     self.data[X_col],
        #     self.data[y_col],
        #     train_size=train_size,
        #     random_state=random_seed,
        #     stratify=self.data[y_col])

        # return X_train, X_test, y_train, y_test
        stratify_by = self.data[y_col] if stratify else None

        train, test = train_test_split(
            self.data,
            train_size=train_size,
            random_state=random_seed,
            stratify=stratify_by)

        return train, test


class ToxicDataset(Dataset):
    _X_COL = 'comment_text_clean'
    _Y_COL = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._texts = []
        self._labels = []

        for idx in range(len(self.data)):
            self._texts.append(self.data.loc[idx, self._X_COL])
            self._labels.append(self.data.loc[idx, self._Y_COL].tolist())

    def __len__(self):
        return len(self.data)

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

    @classmethod
    def get_labels(self):
        return self._Y_COL
