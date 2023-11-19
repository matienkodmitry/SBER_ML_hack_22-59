import joblib
from utility import features_creation_advanced, predict, preprocess_text
import pandas as pd
import os
import re
import streamlit as st
import numpy as np
from utility import features_graph

#PATH_DATA = 'data'
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
tr_mcc_codes = pd.read_csv('mcc_codes.csv', sep=';', index_col='mcc_code')
tr_types = pd.read_csv('trans_types.csv', sep=';', index_col='trans_type')


#transactions = pd.read_csv(os.path.join(PATH_DATA, 'transactions.csv'), index_col='client_id')
#gender_test = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'), index_col='client_id')
#gender_train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'), index_col='client_id')
#transactions_train = transactions.join(gender_train, how='inner')
#transactions_test = transactions.join(gender_test, how='inner')


st.title('Предсказания пола покупателя по транзакциям')



mcc_options = tr_mcc_codes['mcc_description'].unique()
trans_options = tr_types['trans_description'].unique()

# Получаем выбранные пользователем значения
mcc_description = st.selectbox('Категория покупки', mcc_options)
trans_description = st.selectbox('Тип транзакции', trans_options)
amount = st.number_input('Сумма транзакции',value=1000)
day = st.number_input('День недели транзакции', min_value=0, max_value=6)
day_month = st.number_input('День месяца транзакции', min_value=0, max_value=29)
hour = st.number_input('Час совершения транзакции', min_value=0, max_value=23)
night = hour < 6 or hour >= 22
mcc_description_dict = {'trans_description' : trans_description,'mcc_description' : mcc_description,  \
                       'amount' : amount, 'day' : day, 'day_month' : day_month, 'hour' : hour,'night':night}


transactions_test = pd.DataFrame([mcc_description_dict])


transactions_test['mcc_description']= transactions_test['mcc_description'].apply(lambda x: preprocess_text(x, True))
transactions_test['trans_description']= transactions_test['trans_description'].apply(lambda x: preprocess_text(x, True))


data_test = features_creation_advanced(transactions_test, enc_mcc, enc_types, VOCAB_MCC, VOCAB_TYPES)


length = 767 - len(data_test.columns)
for i in range(length):
    column_name = f'new_column_{i}'
    data_test[column_name] = np.nan

btn_predict = st.button('Предсказать')
if btn_predict:
    st.write('Пол:')
    clf, submission = predict(model, data_test, '')
    if submission['probability'][0] >= 0.33:
        st.write(f'Мужчина')
    else:
        st.write('Женщина')
clf, submission = predict(model, data_test, '')
print(submission['probability'][0])
#st.title()
#if submission < 0.5:
