import joblib
from utility import features_creation_advanced, predict, preprocess_text
import pandas as pd
import os
import re

PATH_DATA = './data'
# PATH_DATA = r'C:\Users\bende\Projects\PycharmProjects\SBER_ML_hack_22-59\data'
MODEL_PATH = 'model.pkl'
ENC_MCC_PATH = 'enc_mcc.joblib'
ENC_TYPES_PATH = 'enc_types.joblib'
VOCAB_MCC = 521
VOCAB_TYPES = 171

model = joblib.load(MODEL_PATH)
enc_mcc = joblib.load(ENC_MCC_PATH)
enc_types = joblib.load(ENC_TYPES_PATH)

# Считываем данные
tr_mcc_codes = pd.read_csv(os.path.join(PATH_DATA, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
tr_types = pd.read_csv(os.path.join(PATH_DATA, 'trans_types.csv'), sep=';', index_col='trans_type')

transactions = pd.read_csv(os.path.join(PATH_DATA, 'transactions.csv'), index_col='client_id')
gender_train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'), index_col='client_id')
gender_test = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'), index_col='client_id')
transactions_train = transactions.join(gender_train, how='inner')
transactions_test = transactions.join(gender_test, how='inner')

tr_mcc_codes['mcc_description'] = tr_mcc_codes['mcc_description'].apply(lambda x: preprocess_text(x, True))
tr_types['trans_description'] = tr_types['trans_description'].apply(lambda x: preprocess_text(x, True))

for df in [transactions_test]:
    df['day'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) % 7)
    df['day_month'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) % 30)
    df['hour'] = df['trans_time'].apply(lambda x: re.search(' \d*', x).group(0)).astype(int)
    df['night'] = ~df['hour'].between(6, 22).astype(int)

data_test = transactions_test.groupby(transactions_test.index).apply(
    lambda x: features_creation_advanced(x, enc_mcc, tr_mcc_codes, enc_types, tr_types, VOCAB_MCC, VOCAB_TYPES))

clf, submission = predict(model, data_test, '')
submission.to_csv('result.csv')
