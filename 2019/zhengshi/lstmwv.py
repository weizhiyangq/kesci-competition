# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:21:22 2019

@author: YWZQ
"""
'''

import pandas as pd
import gensim
import numpy as np
test =  pd.read_csv('/home/kesci/input/bytedance/first-round/test.csv',header=None)
test.columns = ['query_id','query','query_title_id','title']
test_query_list = test['query'].values.tolist()
test_title_list =test['title'].values.tolist()
test_all = test_query_list+test_title_list
test_all_list = [i.split(' ') for i in test_all]


train_1qian =  pd.read_csv('/home/kesci/work/train_1qian.csv')
train7bai = train_1qian[:7000000]

train_query_list = train7bai['query'].values.tolist()
train_title_list =train7bai['title'].values.tolist()
train_all = train_query_list+train_title_list
train7bailist = [i.split(' ') for i in train_all]

train_test_list = train7bailist+test_all_list

text_all = test_all + train_all
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_model = TfidfVectorizer(min_df=0,token_pattern=r"(?u)\b\w+\b").fit(text_all)

import pickle
pickle.dump(tfidf_model, open("/home/kesci/work/tfidf_model.pickle", "wb"))
#vectorizer = pickle.load(open("/home/kesci/work/tfidf_model.pickle"), "rb"))

from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


model = word2vec.Word2Vec(train_test_list, hs=1,sg=1,min_count=1,window=3,size=160) 
model.save('/home/kesci/work/w2v_of_muti_txt2.model') 

model.wv.save_word2vec_format('/home/kesci/work/w2v_of_muti_txt2.txt',fvocab='/home/kesci/work/w2v_vocab_of_multi_txt2.txt')#w2v_of_muti_txt.txt为词向量，每行为对应的词以及向量，在语料库中出现频次越多的越靠前。fvocab则是词以及出现的次数，也是频次越多越靠前

from scipy.linalg import norm

print(test[:10])
test.iloc[0,:]['title']


def vector_tfidf_similarity(df):
    def sentence_vector(s):
        s_tfidf = tfidf_model.transform([s])
        s_tfidf_data = list(s_tfidf.data)
        s_tfidf_sum =sum(s_tfidf_data)
        words = s.split(' ')
        v = np.zeros(160)
        len_word = len(words)
        len_tfidf = len(s_tfidf_data)
        if len_word == len_tfidf:
            for i in range(len_word):
                v += model[words[i]]*s_tfidf_data[i]
            v /= s_tfidf_sum
        
        else:
            for word in words:
                v += model[word]
            v /= len(words)
        
        return v
    
    v1, v2 = sentence_vector(df['query']), sentence_vector(df['title'])
#    print('begin similar')
    df['sim'] = np.dot(v1, v2) / (norm(v1) * norm(v2))
    return df


test = test.apply(vector_tfidf_similarity,axis=1)
test_pre = test[['query_id','query_title_id','sim']]
test_pre.to_csv('/home/kesci/work/tfidf_sim_submit.csv',index=None,header=None)
train100 = train7bai[:100]
train100 = train100.apply(vector_tfidf_similarity,axis=1)
print(train100)


def df_w2v(df):
    def sentence_vector(s):
        words = s.split(' ')
        v = np.zeros(160)
        for word in words:
            v += model[word]
        v /= len(words)
        return v 
    v1, v2 = sentence_vector(df['query']), sentence_vector(df['title'])
    v1_list,v2list = list(v1),list(v2)
    df['sequence'] = v1_list+v2list
    return df

test = test.apply(df_w2v,axis=1)
train7bai = train7bai.apply(df_w2v,axis=1)

train7bai_data = np.array(train7bai[['sequence']].values.tolist())
train7bai_label = np.array(train7bai[['label']].values.tolist())

test_data = np.array(test[['sequence']].values.tolist())

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(64))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metric=['acc'])
history = model.fit(train7bai_data,train7bai_label,epoch=5,batch_size=7000,validation_split=0.2,verbose=2)
test_pre = model.predict_proba(test_data)
test[['pre']] =  test_pre

'''
import numpy as np
a=np.array([[1,2,10],[1,2,3],[1,2,1]])
b=np.array([[1,2,3]]).transpose()
c=a*b
print(a)
print(b)
print(c)
print(np.mean(c,axis=0))

d=np.array([6,6,6])
e=np.array([8,8,8])
f=np.array([9,9,9])

g=np.vstack((d,e,f))
h=g*b
print(h)
words = ['a','b','c']
w2v_matrix = np.array([i+'bb'for i in words])
print(w2v_matrix)

aaa=np.array([1,2,3])
print(aaa.sum())