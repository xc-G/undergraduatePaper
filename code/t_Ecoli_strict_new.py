# encoding: utf-8
#! /usr/bin/env python
import imp
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
import csv
f_mark=open('/home/gaoxinchen/GeneClassification/data/EcoliPEC_mark.txt')
#f_mark=open('EcoliPEC_mark.txt') #打开类别标签文件

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


def t_test(yX,k):

	yX_train,yX_test,y_train,y_test = train_test_split(yX,y,test_size=0.2,shuffle=True)

	df_train=DataFrame(yX_train)
	df_test=DataFrame(yX_test)
	#total_row=df.shape[1]-1
	feature_count=4**k*3
	total_features = int(feature_count*0.01)#筛选的特征数目
	#total_features = 5
	TTestResult = {}
	non_essential_set = df_train[df_train.iloc[:, 0] == -1] #非必需基因集
	essential_set = df_train[df_train.iloc[:,0] == 1] #必需基因集

	for i in range(1,(df_train.columns.size)):
		non_essential = non_essential_set.iloc[:,i]
		essential = essential_set.iloc[:,i]
		t_test_result = ss.ttest_ind(non_essential, essential,equal_var=False)
		TTestResult[i] = t_test_result[1]
	
	ResultRank = sorted(TTestResult.items(), key=lambda x:x[1], reverse=False)
	row_id=[]
	for q in range(0,total_features):
		j=ResultRank[q]
		row_id.append(j[0])
	
	DF_train=df_train[row_id]
	DF_test=df_test[row_id]
	X_train=np.array(DF_train)
	X_test=np.array(DF_test)
	
	return X_train,X_test,y_train,y_test

def crosstest(X_train,X_test,y_train,y_test):
	auc_l=[]

	clf = LinearSVC(dual=False,random_state=None,class_weight='balanced')
	clf=clf.fit(X_train,y_train.ravel())
	scores=clf.decision_function(X_test)
	scores=np.array(scores)

	auc=roc_auc_score(y_test,scores)
	auc_l.append(auc)
	print(auc)
	#auc_mean=np.mean(auc_l)
	#print('\n',auc_mean)



def main():
	k=7
	for index in range(0,9):
		featurefile = '/home/gaoxinchen/GeneClassification/data/'+str(k)+'_'+str(index)+'feature.csv'
		#featurefile = str(k)+'_'+str(index)+'feature.csv'
		feature=np.loadtxt(featurefile)
		#print(feature.shape)
		X_train,X_test,y_train,y_test=t_test(feature,k)
		if index == 0:
			L_X_train=X_train
			L_X_test=X_test
		else :
			L_X_train=np.hstack((L_X_train,X_train))
			L_X_test=np.hstack((L_X_test,X_test))

	L_X_train=np.array(L_X_train)
	L_X_test=np.array(L_X_test)
	sample_new=np.vstack((L_X_train,L_X_test))
	crosstest(L_X_train,L_X_test,y_train,y_test)
	


main()





