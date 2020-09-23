# -*- coding = utf-8 -*-
# @time:2020/9/23 8:56
# Author:TC
# @File:random forest.py
# @Software:PyCharm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

wine=load_wine()
clf=DecisionTreeClassifier(random_state=15)
rfc=RandomForestClassifier(random_state=15)

Xtrain,Xtest,Ytrain,Ytest=train_test_split(wine.data,wine.target,test_size=0.3)
clf.fit(Xtrain,Ytrain)
rfc.fit(Xtrain,Ytrain)

score_1=clf.score(Xtest,Ytest)
score_2=rfc.score(Xtest,Ytest)
print('决策数：评分{}'.format(score_1),'随机森林：评分{}'.format(score_2))


#决策树和随机森林在交叉验证下对比
from sklearn.model_selection import cross_val_score
clf=DecisionTreeClassifier(random_state=15)
rfc=RandomForestClassifier(random_state=15)
score_1=cross_val_score(clf,wine.data,wine.target,cv=10)
score_2=cross_val_score(rfc,wine.data,wine.target,cv=10)

plt.plot(range(1,11),score_1,label='Decision Tree')
plt.plot(range(1,11),score_2,label='RandomForest')
plt.legend()
plt.show()

# 随机森林和决策树在十组交叉验证下的效果对比
clf_l = []
rfc_l = []
for i in range(10):
    clf = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    l1 = cross_val_score(clf,wine.data,wine.target,cv=5).mean()
    l2 = cross_val_score(rfc,wine.data,wine.target,cv=5).mean()
    clf_l.append(l1)
    rfc_l.append(l2)
plt.plot(range(1,11),clf_l,label="Decision Tree")
plt.plot(range(1,11),rfc_l,label="Random Forest")
plt.legend()
plt.show()


#n_estimators参数影响
rfc_l=[]
for i in range(200):
    rfc=RandomForestClassifier(n_estimators=i+1)
    rfc_s=cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
    rfc_l.append(rfc_s)
print(max(rfc_l),rfc_l.index(max(rfc_l))+1)
plt.plot(range(1,201),rfc_l,label='随机森林')
plt.legend()
plt.show()

# #使用estimators_属性查看森林中树的情况   可以看单独的一棵树，甚至可以看到参数  决策树是不可以的
# #rfc.estimators_[0].random_state
# for i in range(len(rfc.estimators_)):
#     print(rfc.estimators_[i].random_state)  #每次都是不变的