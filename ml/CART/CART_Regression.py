#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np 
import sys 
import copy 

#定义树结点的基本结构，建树及划分方法
class RT:
    def __init__(self,data,y):
        self.data=data
        self.y = y #该条样本对应的标准值
        self.j = None #最优划分特征的index
        self.s = None #最优划分特征的最优切分点
        self.left = None #左子树
        self.right = None #右子树
        self.isLeaf = True #是否是叶子结点
        self.pred = None #该结点若是叶子结点，则其对应的预测值
        self.grown() #创建RT对象时就建立树
    #最优划分的标准——当前特征划分值来划分样本后得到的均方误差
    def err(self,index):
        c = np.mean(self.y[index])
        return np.mean(np.square([c-self.y[i] for i in range(len(self.y))]))
    #找最优划分的方法
    def bestSlice(self):
        #找到最优的j,s
        minerr = sys.maxsize
        j_best=None
        s_best= None#(self.data[self.sliceIndex[0],0]+self.data[self.sliceIndex[1],0])/2
        for j in range(self.data.shape[1]):#遍历所有特征
            col = np.sort(self.data[:,j]) #一列数据取划分点时先排序，否则切分点不能完全将数据划分开
            s_ready = (col[1:]+col[:-1])/2 #所有数据该特征的预备的切分点
            for s in s_ready:
                R1,R2 = [],[]
                R1 = [index for index in range(len(col)) if col[index] <s ]
                R2 = [index for index in range(len(col)) if col[index] >=s]
                err = self.err(R1)+self.err(R2)
                if minerr > err:
                    j_best = j 
                    s_best = s
        return j_best,s_best
    #建立树
    def grown(self):
        self.pred = np.mean(self.y)
        nums = self.data.shape[0]
        if nums<2:
            return
        j,s = self.bestSlice()
        self.j = j 
        self.s = s 
        self.isLeaf = False
        leftIndex ,rightIndex =[],[]
        for i in range(nums):
        #for i in self.sliceIndex:
            if self.data[i,j] < s:
                leftIndex.append(i)
            else: rightIndex.append(i)
        self.left = RT(self.data[leftIndex,:],self.y[leftIndex]) #用划分后的数据构建子树
        self.right =RT(self.data[rightIndex,:],self.y[rightIndex])

#预测
def prediction(T,x):
    if T.isLeaf:
        return T.pred
    else:
        if x[T.j] < T.s:
            return prediction(T.left,x)
        else: return prediction(T.right,x)
#算均方根误差 及 对比预测数据和标签值
def rootMeanSquareError(T,X,y):
    pred = [prediction(T,X[i,:]) for i in range(X.shape[0])]
    df = pd.DataFrame({'pred':pred,'label':y})
    return np.sqrt(np.mean(np.square([pred[i]-y[i] for i in range(len(y))])))

data = pd.read_csv("./housing.csv")
y = data['MEDV'].values #标签值
dv_train=data.drop(columns='MEDV',axis=1).iloc[0:50,:].values #取所有数据要跑很久，所以只取了部分数据
dv_test=data.drop(columns='MEDV',axis=1).iloc[50:100,:].values #取所有数据要跑很久，所以只取了部分数据

tree = RT(dv_train,y) #建立一颗CART回归树


RMSE = rootMeanSquareError(tree,dv_test,y[50:100]) #算均方根误差 及 对比预测数据和标签值
print("RMSE is {0}".format(RMSE)) 

