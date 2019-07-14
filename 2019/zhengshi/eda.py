# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 21:57:56 2019

@author: YWZQ
"""

import pandas as pd
train1qian = pd.read_csv('/home/kesci/work/train_1qian.csv')
test =  pd.read_csv('/home/kesci/input/bytedance/first-round/test.csv',header=None)
test.columns = ['query_id','query','query_title_id','title']

train1qian['title_list'] = train1qian['title'].map(lambda x:x.split(' '))
train1qian['len_title'] = train1qian['title_list'].map(lambda x:len(x))
train1qian['query_list'] = train1qian['query'].map(lambda x:x.split(' '))
train1qian['len_query'] = train1qian['query_list'].map(lambda x:len(x))
train1qian['query_title_lendiff'] = train1qian['len_title']-train1qian['len_query']

print(train1qian['len_title'] )
test['title_list'] = test['title'].map(lambda x:x.split(' '))
test['len_title'] = test['title_list'].map(lambda x:len(x))
test['query_list'] = test['query'].map(lambda x:x.split(' '))
test['len_query'] = test['query_list'].map(lambda x:len(x))
test['query_title_lendiff'] = test['len_title']-test['len_query']

import seaborn as sns
import matplotlib.pyplot as plt

'''
正常的情况下，训练集和测试集的特征分布应该一致，这样根据训练集训练得到的模型才对测试集适用。
一致的分布意味着，取值范围一致，概率密度核kde分布曲线类似。
'''
import lightgbm as lgb
import gc
def plot_kde(train,test,col,values=True):
    fig,ax = plt.subplots(1,4,figsize=(12,6))
    sns.kdeplot(train[col][train['label']==0],color='g',ax=ax[0])
    sns.kdeplot(train[col][train['label']==1],color='y',ax=ax[0])
    sns.kdeplot(train[col][train['label']==-1],color='r',ax=ax[0])
    sns.kdeplot(train[col],color='g',ax=ax[1])
    sns.kdeplot(test[col],color='y',ax=ax[2])
    
    sns.kdeplot(train[col],color='g',ax=ax[3])
    sns.kdeplot(test[col],color='y',ax=ax[3])
    plt.show()
    del train,col,test
    gc.collect()
cols = ['len_title','len_query','query_title_lendiff']
for col in cols:
    plot_kde(train1qian,test,col)
    
train_pos = train1qian[:5000000]
print(train_pos[train_pos['label']==1].shape)

train_pos['title_list'] = train_pos['title'].apply(lambda x:x.split(' '))
print(train_pos.head())

#注意保存为csv文件的话，dataframe的数据类型可能不被保存，如list形式的结构会变为str形式，可以用hdf文件保存