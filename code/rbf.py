from Bio import SeqIO
import os
import numpy as np
from sklearn.feature_selection import RFE 
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve,auc
from pandas import DataFrame
from scipy import stats as ss
from sklearn.model_selection import KFold,StratifiedKFold
import concurrent.futures
from sklearn.svm import SVC
import time

start = time.perf_counter()

f_mark=open('../data/EcoliPEC_mark.txt') 
mark_dic={}#字典的键是基因名，值是类
for line_1 in f_mark:#逐行读取
    line_1=line_1.strip()
    mark_list=[]
    mark_list=line_1.split("	")	
    mark_dic[mark_list[0]]=mark_list[1]
mark_l=[]
for mark_values in mark_dic.values():
    mark_l.append([int(mark_values)])
y=np.array(mark_l)


def t_test():
    K=6
    lamda=4
    inf='k'+str(K)+'_lamda'+str(lamda)
    road='../data/'+inf+'.csv'
    X=np.loadtxt(road)
    yX=np.hstack((mark_l,X))
    folder = StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    
    auc_l=[]
    for train,test in folder.split(yX,y):
        yX_train=yX[train]
        y_train=y[train]
        yX_test=yX[test]
        y_test=y[test]
        df_train=DataFrame(yX_train)
        df_test=DataFrame(yX_test)
        np.seterr(divide='ignore',invalid='ignore')
        TTestResult = {}
        non_essential_set = df_train[df_train.iloc[:, 0] == -1] #非必需基因集
        essential_set = df_train[df_train.iloc[:,0] == 1] #必需基因集

        for i in range(1,(df_train.columns.size)):
            non_essential = non_essential_set.iloc[:,i]
            essential = essential_set.iloc[:,i]
            
            t_test_result = ss.ttest_ind(non_essential, essential,equal_var=False)
            TTestResult[i] = t_test_result[1]
        
        ResultRank = sorted(TTestResult.items(), key=lambda x:x[1], reverse=False)
        p=1e-3
        count=[]
        i=0 
        row_id=[]
        while ResultRank[i][1]<p:
            j=ResultRank[i]
            row_id.append(j[0])
            i=i+1
        count.append(i)
        
        DF_train=df_train[row_id]
        DF_test=df_test[row_id]
        X_train=np.array(DF_train)
        X_test=np.array(DF_test)
        clf = SVC(C=1.0,gamma=0.5,kernel='rbf',random_state=None,class_weight='balanced')
        #clf = LinearSVC(dual=False,random_state=None,class_weight='balanced')
        clf=clf.fit(X_train,y_train.ravel())
        scores=clf.decision_function(X_test)
        #scores=clf.predict_proba(X_test)[:,1]

        scores=np.array(scores)
        
        auc=roc_auc_score(y_test,scores)
        auc_l.append(auc)
        print('auc ',auc)
    auc_mean=np.mean(auc_l)
    print('auc_mean ',auc_mean)
    end = time.perf_counter()
    print('running_time',end-start)
t_test()
