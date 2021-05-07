import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


class DataIngest:
    def __init__(self):
        self._data = ''
    
    def read_data(self, file_path, source_type):
        source_type = source_type.lower()
        if source_type == 'db':
            # TODO implement DB
            pass
        elif source_type == 'csv':
            return pd.read_csv(file_path)
        else:
            raise Exception('Unknown file format. Please check')


class DataPipeline:
    def __init__(self):
        self._data = ''

    def clean_data(self, data):
        self._data = data
        self._data.drop('id', axis=1, inplace=True)
        self._data['toxic_cum_sum'] = self._data.iloc[:, 1:].sum(axis=1)
        self._data['is_negative'] = self._data['toxic_cum_sum']\
            .apply(lambda x: 1 if x >= 1 else 0)

        # df['sentence_count'] = df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
        self._data['sentence_count'] = self._data["comment_text"]\
            .apply(self._get_sentence_count)

        # does toxic comments have longer word count?
        self._data['count_word'] = self._data["comment_text"]\
            .apply(lambda x: len(str(x).split()))

        # does toxic comments have more unique words?
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
        print ('add_stopwords:', add_stopwords, type(add_stopwords))
        tok_clean_comments = []
        if add_stopwords is None:
            stop_words = set(stopwords.words('english'))
        elif type(add_stopwords) == list:
            for i in add_stopwords:
                stop_words.add(i)
        else:
            raise Exception('Unknown input. Please check.')
        lemmatizer = WordNetLemmatizer()
        for i in range(self._data.shape[0]):
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


    def split_data(self, X_col, y_col, test_size=0.2, random_seed=42):
        if type(X_col) != list and type(y_col) != list:
            raise Exception(f'Expecting a list of column names, \
                received X_col: {type(X_col)} and y_col: {type(y_col)}. Please check')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X=self._data[X_col],
            y=self._data[y_col],
            test_size=test_size,
            random_state=random_seed,
            stratify=self._data[y_col])

        return X_train, X_test, y_train, y_test

    
