# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:50:35 2019

@author: YWZQ
"""

import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier as SGD      #随机梯度下降算法
from sklearn.naive_bayes import MultinomialNB              #朴素贝叶斯分类器
from sklearn.neighbors import KNeighborsClassifier         #K近邻分类器
from sklearn.svm import SVC                                #svm分类器svc
#from sklearn.ensemble import RandomForestClassifier        #随机森林分类器
from sklearn.linear_model import LogisticRegression as LR  #逻辑回归分类器

from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV     #概率校准函数

import warnings;warnings.filterwarnings('ignore')

train = pd.read_csv("data/train.csv", lineterminator='\n', header=0)
train['label'] = train['label'].map({'Negative':0, 'Positive': 1})
#print(train.isnull().sum())
test = pd.read_csv("data/20190610_test.csv", lineterminator='\n', header=0)
#print(test.isnull().sum())


words = []
for _ in train['review'].values:
    words.append(' '.join(WordPunctTokenizer().tokenize(_)))
train_data = words
train_label = np.array(train['label'].values, dtype='int8')

words = []
for _ in test['review'].values:
    words.append(' '.join(WordPunctTokenizer().tokenize(_)))
test_data = words

ngram = 2
vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, ngram), max_df=0.9)
corpus_all = train_data + test_data
vectorizer.fit(corpus_all)
dense = False
if dense == True:
    corpus_all = vectorizer.transform(corpus_all).todense()
    name = 'dense'
else:
    corpus_all = vectorizer.transform(corpus_all)
    name = ''
print(corpus_all.shape)

lentrain = len(train_data)
train_data = corpus_all[:lentrain]
test_data = corpus_all[lentrain:]
print(train_data[0])
# model training and test
folds = StratifiedKFold(n_splits=30, shuffle=False, random_state=0)
predictions = np.zeros(test_data.shape[0])
score_label = np.zeros(train_data.shape[0])

aucs = []

for fold_, (train_index, test_index) in enumerate(folds.split(train_data, train_label)):
    cv_train_data, cv_train_label= train_data[train_index], train_label[train_index]
    cv_test_data, cv_test_label = train_data[test_index], train_label[test_index]
    
    #1.随机梯度下降做分类
    #train_model = SGD(alpha=0.00001, penalty='l2', tol=10000, shuffle=True, loss='log')
    #2.朴素贝叶斯分类器
    train_model = MultinomialNB()
    #3.k近邻
#    train_model = KNeighborsClassifier()
    #4.svm分类器svc
#    train_model = SVC(gamma='auto', probability=True)   #速度慢，结果差
    #5.随机森林
#    train_model = RandomForestClassifier()              #速度慢，效果差
    #6.逻辑回归分类器
#    train_model = LR(solver='lbfgs')
    
    #线性svm搭配cal概率校准
#    train_model = svm.LinearSVC()
#    model = CalibratedClassifierCV(train_model,cv=5)
    model = train_model

        
    model.fit(cv_train_data, cv_train_label)     
    score_label[test_index] = model.predict_proba(cv_test_data)[:, 1]    
    auc = metrics.roc_auc_score(cv_test_label, model.predict_proba(cv_test_data)[:, 1])
    predictions += model.predict_proba(test_data)[:, 1] / folds.n_splits
    aucs.append(auc)
    print("Fold :{}".format(fold_ + 1),"auc score: %.5f" % auc)

print('Mean auc', np.mean(aucs))
print('All  auc',metrics.roc_auc_score(train_label,score_label))


predictions = pd.DataFrame(predictions)
id = pd.DataFrame(np.arange(1, len(predictions) + 1))
data = pd.concat([id, predictions], axis=1)

data.to_csv('sub/{}{}_{}.csv'.format((str(train_model)[0:3]),name,str(np.mean(aucs))[0:5]), header=['ID', 'Pred'], index=False)








