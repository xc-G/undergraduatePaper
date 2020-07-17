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

f_mark=open('../data/EcoliPEC_mark.txt') #打开类别标签文件
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
def base2digital(base):
	if base == 'a':
		return 0
	elif base == 'c':
		return 1
	elif base == 'g':
		return 2
	else:
		return 3

def seq2num(sequence):
	index = 0
	for i in range(0, len(sequence)):
		index = 4*index + int(sequence[i])
	return index

def subseqM2one(i,k,lamda,sequence):
	subseq=sequence[i:i+k-1]+sequence[i+k-2+lamda+1]
	FrequencyIndex=seq2num(subseq)
	return FrequencyIndex

def workSingleBase():
	Feature_list=[0]*12
	for record in SeqIO.parse(open('../data/PECDNAsequence.txt'),'fasta'):
		seq=str(record.seq)
		lengthSeq=len(seq)
		sequenceNUM_list=[]
		for i in range(0,lengthSeq):
			base2num=base2digital(seq[i])
			sequenceNUM_list.append(base2num)
		firstPhaseIndex = range(0, lengthSeq, 3)
		secondPhaseIndex = range(1, lengthSeq, 3)
		thirdPhaseIndex = range(2, lengthSeq, 3)
		firstPhaseFeature = [0]*4
		secondPhaseFeature = [0]*4
		thirdPhaseFeature = [0]*4
		for i1 in firstPhaseIndex:
			firstPhaseFeature[sequenceNUM_list[i1]]+=1
		firstPhaseFeature=[f1/(((i1-0))/3+1) for f1 in firstPhaseFeature]
		for i2 in secondPhaseIndex:
			secondPhaseFeature[sequenceNUM_list[i2]]+=1
		secondPhaseFeature=[f2/(((i2-1))/3+1) for f2 in secondPhaseFeature]
		for i3 in thirdPhaseIndex:
			thirdPhaseFeature[sequenceNUM_list[i3]]+=1
		thirdPhaseFeature=[f3/(((i3-2))/3+1) for f3 in thirdPhaseFeature]
		FeatureALL=firstPhaseFeature+secondPhaseFeature+thirdPhaseFeature
		Feature_list=np.vstack((Feature_list,FeatureALL))
	Feature_list=np.delete(Feature_list,0,0)
	return Feature_list

def workMultipleBase(K,L):
	Feature_list=[[0,0]]*4144
	Feature_list_1=workSingleBase()
	for k in range(2,K):
		for lamda in range(0,L):
			feature_list=[0]*(4**k)*3
			for record in SeqIO.parse(open('../data/PECDNAsequence.txt'),'fasta'):
				seq=str(record.seq)
				lengthSeq=len(seq)
				sequenceNUM_list=[]
				for i in range(0,lengthSeq):
					base2num=base2digital(seq[i])
					sequenceNUM_list.append(base2num)
				sequenceNUM_list=''.join(str(s) for s in sequenceNUM_list)
				firstPhaseIndex = range(0, lengthSeq, 3)
				secondPhaseIndex = range(1, lengthSeq, 3)
				thirdPhaseIndex = range(2, lengthSeq, 3)
				firstPhaseFeature = [0]*(4**k)
				secondPhaseFeature = [0]*(4**k)
				thirdPhaseFeature = [0]*(4**k)
				FeatureALL=[]
				for i1 in firstPhaseIndex:
					if i1+k-2+lamda+1<lengthSeq:
						FrequencyIndex=subseqM2one(i1,k,lamda,sequenceNUM_list)
						firstPhaseFeature[FrequencyIndex]+=1
				firstPhaseFeature=[f1/(((i1-0))/3+1) for f1 in firstPhaseFeature]
				for i2 in secondPhaseIndex:
					if i2+k-2+lamda+1<lengthSeq:
						FrequencyIndex=subseqM2one(i2,k,lamda,sequenceNUM_list)
						secondPhaseFeature[FrequencyIndex]+=1
				secondPhaseFeature=[f2/(((i2-1))/3+1) for f2 in secondPhaseFeature]
				for i3 in thirdPhaseIndex:
					if i3+k-2+lamda+1<lengthSeq:
						FrequencyIndex=subseqM2one(i3,k,lamda,sequenceNUM_list)
						thirdPhaseFeature[FrequencyIndex]+=1
				thirdPhaseFeature=[f3/(((i3-2))/3+1) for f3 in thirdPhaseFeature]
				FeatureALL=firstPhaseFeature+secondPhaseFeature+thirdPhaseFeature
				feature_list=np.vstack((feature_list,FeatureALL))
			feature_list=np.delete(feature_list,0,0)
			Feature_list=np.hstack((Feature_list,feature_list))
	Feature_list=np.hstack((Feature_list,Feature_list_1))
	Feature_list=np.delete(Feature_list,[0,1],1)
	FeatureAndLabels=np.hstack((mark_l,Feature_list))
	shape=Feature_list.shape
	return Feature_list,FeatureAndLabels,shape

def t_test():
	#for K in range(2,7):
		#for lamda in range(1,K+1):
	K=5
	lamda=2
	if K == 1:
		X=workSingleBase() 
	else:
		X,yX,shape=workMultipleBase(K+1,lamda+1)
		inf='shape'+'_k:'+str(K)+'_lamda:'+str(lamda)
		print(inf)
		print(shape)

	df_yX=DataFrame(yX)
	np.seterr(divide='ignore', invalid='ignore')
	total_features=int(shape[1]*0.01) #筛选的特征数目
	TTestResult = {}
	non_essential_set = df_yX[df_yX.iloc[:, 0] == -1] #非必需基因集
	essential_set = df_yX[df_yX.iloc[:,0] == 1] #必需基因集

	for i in range(1,(df_yX.columns.size)):
		non_essential = non_essential_set.iloc[:,i]
		essential = essential_set.iloc[:,i]
		t_test_result = ss.ttest_ind(non_essential, essential,equal_var=False)
		TTestResult[i] = t_test_result[1]

	ResultRank = sorted(TTestResult.items(), key=lambda x:x[1], reverse=False)
	'''
	p=1e-3
	count=[]
	i=0
	'''
	row_id=[]
	'''
	while ResultRank[i][1]<p:
		j=ResultRank[i]
		row_id.append(j[0])
		i=i+1	
	
	count.append(i)
	'''
	for q in range(0,total_features):
		j=ResultRank[q]
		row_id.append(j[0])

	DF_X=df_yX[row_id]
	X=np.array(DF_X)
	folder = StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
	auc_l=[]
	for train,test in folder.split(X,y):
		X_train=X[train]
		X_test=X[test]
		y_train=y[train]
		y_test=y[test]
		clf = LinearSVC(dual=False,random_state=None,class_weight='balanced')
		clf=clf.fit(X_train,y_train.ravel())
		scores=clf.decision_function(X_test)
		scores=np.array(scores)	
		auc=roc_auc_score(y_test,scores)
		auc_l.append(auc)
		print('auc: ',auc)
	auc_mean=np.mean(auc_l)
	print('auc_mean: ',auc_mean)
t_test()
