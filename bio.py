'''
BIO 표현

B: Begin   개체명이 시작되는 부분
I: Inside  개체명의 내부부분
O: Outside 개체명이 아닌부분
'''

import pandas as pd

FILE = ''

df_train = pd.read_csv('train_data_bert', names=['src', 'tar'], sep="\t")
print(df_train)

df_train.reset_index()
print(df_train)