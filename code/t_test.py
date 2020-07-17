#!/usr/bin/env python
# -*- coding: utf-8 -*

import pandas as pd
from scipy import stats as ss

InputFile = '/home/wangshuxuan/MasterSubject/TestOutput/zheng_w3.csv'

def main(FileName):
    TTestResult = {}
    df = pd.read_csv(InputFile, header=None)
    non_essential_set = df[df.iloc[:, 0] == -1]
    essential_set = df[df.loc[:, 0] == 1]
    for i in range(1, (df.columns.size)):
        non_essential = non_essential_set.iloc[:, i]
        essential = essential_set.iloc[:, i]
        t_test_result = ss.ttest_ind(non_essential, essential, equal_var=False)
        TTestResult[i]=t_test_result[1]
        print(str(i)+'\t'+str(float(t_test_result[0]))+'\t'+str(float(t_test_result[1])))

    return TTestResult


if __name__ == '__main__':
    Result = main(InputFile)
    ResultRank = sorted(Result.items(), key=lambda x:x[1], reverse=False)
    with open('/home/wangshuxuan/MasterSubject/OutputFile/TTestZheng_w3.txt', 'w+') as f:
        for j in ResultRank:
            f.write(str(j[0])+'\t'+str(j[1])+'\n')
