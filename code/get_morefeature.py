# encoding: utf-8
#! /usr/bin/env python
import imp
from Bio import SeqIO
import os
import numpy as np
import pandas as pd
#f_mark=open('/home/gaoxinchen/GeneClassification/data/EcoliPEC_mark.txt') #打开类别标签文件
f_mark=open('EcoliPEC_mark.txt')
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

def workMultipleBase(k,L):
	
	
	for lamda in range(0,L):
		feature_list=[0]*(4**k)*3
		Feature_list=[[0,0]]*4144
		#for record in SeqIO.parse(open('/home/gaoxinchen/GeneClassification/data/PECDNAsequence.txt'),'fasta'):
		for record in SeqIO.parse(open('PECDNAsequence.txt'),'fasta'):
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
			for i3 in firstPhaseIndex:
				if i3+k-2+lamda+1<lengthSeq:
					FrequencyIndex=subseqM2one(i3,k,lamda,sequenceNUM_list)
					thirdPhaseFeature[FrequencyIndex]+=1
			thirdPhaseFeature=[f3/(((i3-2))/3+1) for f3 in thirdPhaseFeature]
			FeatureALL=firstPhaseFeature+secondPhaseFeature+thirdPhaseFeature
			feature_list=np.vstack((feature_list,FeatureALL))
		feature_list=np.delete(feature_list,0,0)	
		Feature_list=np.hstack((Feature_list,feature_list))
		
		Feature_list=np.delete(Feature_list,[0,1],1)
		FeatureAndLabels=np.hstack((mark_l,Feature_list))
		#filename = '/home/gaoxinchen/GeneClassification/data/'+str(k)+'_'+str(lamda)+'feature.csv'
		filename = str(k)+'_'+str(lamda)+'feature.csv'
		print(Feature_list.shape)
		np.savetxt(filename,FeatureAndLabels)



workMultipleBase(2,1+1)



	