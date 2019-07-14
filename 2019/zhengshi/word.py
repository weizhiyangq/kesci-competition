# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 21:45:26 2019

@author: YWZQ
"""
"""
kesci上运行的一些指令
"""
#!killall python

#!ls /home/kesci/work
# 查看个人持久化工作区文件
#!ls /home/kesci/work/

# 查看当前kernerl下的package
#!pip list --format=columns

# 显示cell运行时长
#%load_ext klab-autotime



import pandas as pd
#import gensim

test =  pd.read_csv('/home/kesci/input/bytedance/first-round/test.csv',header=None)
test.columns = ['query_id','query','query_title_id','title']
test_query_list = test['query'].values.tolist()
test_title_list =test['title'].values.tolist()
test_all = test_query_list+test_title_list

test_all_list = [i.split() for i in test_all]

print(len(test_all))
print(test_all[:5])

train_1qian =  pd.read_csv('/home/kesci/work/train_1qian.csv')
train_5bai_pos = train_1qian[:2500000]
train_5bai_neg = train_1qian[-2500000:]
train_5bai = pd.concat([train_5bai_pos,train_5bai_neg])

train_query_list = train_1qian['query'].values.tolist()
train_title_list =train_1qian['title'].values.tolist()
train_all = train_query_list+train_title_list

print(train_all[-10:])
print(len(train_all))

text_all = test_all + train_all

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_model = TfidfVectorizer(min_df=0,max_df=0.999,token_pattern=r"(?u)\b\w+\b").fit(text_all)

print(len(tfidf_model.vocabulary_))
str_result = tfidf_model.transform(['1912 4371 27 22 42 55 748 74 4508 30 37'])
print(list(str_result.data))

import pickle
pickle.dump(tfidf_model, open("/home/kesci/work/tfidf_model.pickle", "wb"))
tfidf_model = pickle.load(open("/home/kesci/work/tfidf_model.pickle", "rb"))

del train_query_list
del train_title_list
train1qianlist = [i.split() for i in train_all]

train_test_list = train1qianlist+test_all_list

print(len(train_test_list))

from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


model = word2vec.Word2Vec(train_test_list, hs=1,sg=1,min_count=1,window=3,workers=4,size=150,seed=16,iter=14) 
model.save('/home/kesci/work/w2v_iter14_150.model') 
#model.wv.save_word2vec_format('/home/kesci/work/w2v_iter8.txt',fvocab='/home/kesci/work/w2v_iter8_txt2.txt')#w2v_of_muti_txt.txt为词向量，每行为对应的词以及向量，在语料库中出现频次越多的越靠前。fvocab则是词以及出现的次数，也是频次越多越靠前
print('save ok!')


model.similarity('660','8240')

from scipy.linalg import norm
from gensim.models import word2vec
import logging
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


test =  pd.read_csv('/home/kesci/input/bytedance/first-round/test.csv',header=None)
test.columns = ['query_id','query','query_title_id','title']


model = word2vec.Word2Vec.load('/home/kesci/work/w2v_iter14_150.model') 

model.similarity('660','8240')

'''
#运行应该没有new_vector_tfidf_similarity快
import numpy as np
def vector_tfidf_similarity(df):
    def sentence_vector(s):
        #s_tfidf = tfidf_model.transform([s])
        #s_tfidf_data = list(s_tfidf.data)
        #s_tfidf_sum =sum(s_tfidf_data)
        words = s.split(' ')
        v = np.zeros(200)
        #len_word = len(words)
        #len_tfidf = len(s_tfidf_data)
        #if len_word == len_tfidf:

        for word in words:
            v += model[word]
        v /= len(words)
    
        return v
    
    v1, v2 = sentence_vector(df['query']), sentence_vector(df['title'])
#    print('begin similar')
    df['sim'] = np.dot(v1, v2) / (norm(v1) * norm(v2))
    return df
'''

#理论上运行快，因为矩阵运算
import numpy as np
voca = tfidf_model.vocabulary_
#print('1868' in voca.keys())
def new_vector_tfidf_similarity(df):
    def sentence_vector(s):
        '''
        s_tfidf = np.array(tfidf_model.transform([s]).todense())[0]
        s_tfidf_data =np.array([s_tfidf[voca[i]] for i in s.split()])
        s_tfidf_sum =sum(s_tfidf_data)
        words = s.split(' ')
        v = np.zeros(150)
        for i in range(len(words)):
            v += np.array(model[words[i]])*s_tfidf_data[i]
        v /= s_tfidf_sum
        '''
        
        words = s.split()
        s_tfidf = np.array(tfidf_model.transform([s]).todense())[0]
        s_tfidf_sum = s_tfidf.sum()
        s_tfidf_matrix =np.array([[s_tfidf[voca[i]] for i in words]]).transpose()
        w2v_matrix = np.array([model[i]for i in words])
        v=np.sum(w2v_matrix*s_tfidf_matrix,axis=0)/s_tfidf_sum
        
        return v
    
    v1, v2 = sentence_vector(df['query']), sentence_vector(df['title'])
#    print('begin similar')
    df['sim'] = np.dot(v1, v2) / (norm(v1) * norm(v2))
    return df


s = test.iloc[609,3]
s.split()
#print(model['1868'])



test = test.apply(new_vector_tfidf_similarity,axis=1)
test_pre = test[['query_id','query_title_id','sim']]
test_pre.to_csv('/home/kesci/work/submit_14_220_tfidf.csv',index=None,header=None)
print('save test pre ok!')


train_5bai = train_5bai.apply(new_vector_tfidf_similarity,axis=1)
train_5bai_sim = train_5bai[['query_id','query_title_id','sim']]
train_5bai_sim.to_csv('/home/kesci/work/train_5bai_sim.csv',index=None,header=None)
print('save train_5bai_sim pre ok!')

print(test_pre[:10])

test_pre_read = pd.read_csv('/home/kesci/work/submit_14_220_tfidf.csv',header=None)
print(test_pre_read.head(50))



import numpy as np
import gc
voca = tfidf_model.vocabulary_
def df_vec_save(df):
    def sentence_vector(s):
        words = s.split()
        s_tfidf = np.array(tfidf_model.transform([s]).todense())[0]
        s_tfidf_sum = s_tfidf.sum()
        s_tfidf_matrix =np.array([[s_tfidf[voca[i]] for i in words]]).transpose()
        w2v_matrix = np.array([model[i]for i in words])
        v=np.sum(w2v_matrix*s_tfidf_matrix,axis=0)/s_tfidf_sum
        return v
   
    #df['query'] = np.around(sentence_vector(df['query']),decimals=5)
    #df['title'] = np.around(sentence_vector(df['title']),decimals=5)
    #df['que_tit_conc'] = np.hstack((np.around(sentence_vector(df['query']),decimals=5),np.around(sentence_vector(df['title']),decimals=5)))
    df_series = pd.Series(np.hstack((np.around(sentence_vector(df['query']),decimals=4),np.around(sentence_vector(df['title']),decimals=4))))
    #df.drop(['B', 'C'], axis=1)

    return df_series


train2 = train_5bai[2500000:]
train2 = train2.apply(df_vec_save,axis=1)
print(train2.head())
#testsave_vec = test['que_tit_conc'].apply(lambda x:pd.Series(x))

#testsave_id = test[['query_id']]
#testsave_vec = pd.concat([testsave_id,testsave_vec],axis=1)
train2.to_csv('/home/kesci/work/train2_vec_save.csv.gz',compression='gzip',index=False)
print('train2 vec save ok!')


test3 = test[3500000:]
test3 = test3.apply(df_vec_save,axis=1)
print(test3.head())
#testsave_vec = test['que_tit_conc'].apply(lambda x:pd.Series(x))

#testsave_id = test[['query_id']]
#testsave_vec = pd.concat([testsave_id,testsave_vec],axis=1)
test3.to_csv('/home/kesci/work/test3_vec_save.csv.gz',compression='gzip',index=False)
print('test3 vec save ok!')















































