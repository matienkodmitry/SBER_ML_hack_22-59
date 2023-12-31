import pandas as pd
import numpy as np
import xgboost as xgb
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import re
import streamlit as st


def features_graph(model_cb):
    features = model_cb.get_feature_importance(prettified=True)
    st.write(features)
    features = features.sort_values('Importances')
    ax = features.plot.barh(x='Feature Id', y='Importances')
    return ax.figure

nltk.download('stopwords')
none_trans_type = {2446, 2456, 2460, 4096, 4097}


def features_creation_advanced(x, enc_mcc, enc_types, vocab_mcc, vocab_types):
    features = []
    # print(np.concatenate([tr_mcc_codes.loc[i, 'mcc_description'] for i in x['mcc_code']]).reshape(-1, 1))
    #features.append(pd.DataFrame(np.mean(enc_mcc.transform(x['mcc_description'])).reshape(1, -1),columns=[f'onehot_mcc_{i}' for i in range(vocab_mcc)]))

    features.append(pd.DataFrame(np.mean(enc_mcc.transform(
        np.concatenate(x['mcc_description']).reshape(-1, 1)),
        axis=0).reshape(1, -1),
                                 columns=[f'onehot_mcc_{i}' for i in range(vocab_mcc)]))

    features.append(pd.DataFrame(np.mean(enc_types.transform(np.concatenate(
        x['trans_description']).reshape(-1, 1)),
                                         axis=0).reshape(1, -1),
                                 columns=[f'onehot_type_{i}' for i in range(vocab_types)]).reindex(features[0].index))

    features.append(pd.DataFrame(x['day_month'].value_counts(normalize=True).add_prefix('day_month_')).rename(
        columns={'proportion': 0}).transpose().reindex(features[0].index))

    features.append(pd.DataFrame(x['hour'].value_counts(normalize=True).add_prefix('hour_')).rename(
        columns={'proportion': 0}).transpose().reindex(features[0].index))

    features.append(pd.DataFrame(x['night'].value_counts(normalize=True).add_prefix('night_')).rename(
        columns={'proportion': 0}).transpose().reindex(features[0].index))

    features.append(pd.DataFrame(x[x['amount'] > 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count']) \
                                 .add_prefix('positive_transactions_')).rename(
        columns={'amount': 0}).transpose().reindex(features[0].index))

    features.append(pd.DataFrame(x[x['amount'] < 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count']) \
                                 .add_prefix('negative_transactions_')).rename(
        columns={'amount': 0}).transpose().reindex(features[0].index))

    # print(pd.concat((features[0], features[1].reindex(features[0].index)), axis=1))
    return pd.concat(features, axis=1)


def predict(clf, test, path_feature_names):
    feature_names = get_feature_names(path_feature_names)
    y_pred = clf.predict(xgb.DMatrix(test.values, feature_names=feature_names))
    submission = pd.DataFrame(index=test.index, data=y_pred, columns=['probability'])
    return clf, submission


def get_feature_names(path: str = ''):
    res = []
    with open(path + "feature_names.txt", "r") as output:
        for line in output.readlines():
            res.append(line.strip('\n'))
    return res

stopwords = stopwords.words('russian')
stemmer_rus = SnowballStemmer('russian')
stemmer_eng = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')
tags = r"@\w*"


def preprocess_text(sentence, stem=False):
    sentence = [re.sub(tags, "", sentence)]
    text = []
    for word in sentence:

        if word not in stopwords:
            word_rus = re.findall(r'[А-я]+', word)
            word_eng = re.findall(r'[A-z]+', word)
            if stem:
                if word_rus and word_eng:
                    text += [stemmer_rus.stem(' '.join(word_rus)).lower()] + [
                        stemmer_eng.stem(' '.join(word_eng)).lower()]
                elif word_rus:
                    text.append(stemmer_rus.stem(' '.join(word_rus)).lower())
                elif word_eng:
                    text.append(stemmer_eng.stem(' '.join(word_eng)).lower())
            else:
                text.append(word.lower())
    return np.array(tokenizer.tokenize(" ".join(text)), dtype=str)
