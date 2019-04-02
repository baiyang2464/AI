#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np 
import sys 
import copy 

#定义树结点的基本结构，建树及划分方法
class CT:
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
    #基尼系数
    def gini(self,index):
        #基尼指数越大，被被划分后的集合不确定性越大
        p = len(index)/len(self.y)
        return 2*p*(1-p)
    #找最优划分的方法
    def bestSlice(self):
        #找到最优的j,s
        mingini = sys.maxsize
        j_best=None
        s_best= None#(self.data[self.sliceIndex[0],0]+self.data[self.sliceIndex[1],0])/2
        for j in range(self.data.shape[1]):#遍历所有特征
            col = np.sort(self.data[:,j]) #一列数据取划分点时先排序，否则切分点不能完全将数据划分开
            s_ready = (col[1:]+col[:-1])/2 #所有数据该特征的预备的切分点
            for s in s_ready:
                R1 =[]
                R1 = [index for index in range(len(col)) if col[index] <s ]
                gini = self.gini(R1)
                if mingini > gini:
                    j_best = j 
                    s_best = s
        return j_best,s_best
    def findMode(self,nums):
        nums=np.sort(nums)
        maxlen = 0
        count =1
        val =nums[0]
        for i in range(1,len(nums)):
            if nums[i] == nums[i-1]:
                count+=1
                if count > maxlen:
                    maxlen = count
                    count =1
                    val = nums[i]
        return val
    
    #建立树
    def grown(self):
        tmpy = copy.deepcopy(self.y)
        self.pred = self.findMode(tmpy)
        nums = self.data.shape[0]
        if nums<2:
            return
        j,s = self.bestSlice()
        self.j = j 
        self.s = s 
        
        leftIndex ,rightIndex =[],[]
        for i in range(nums):
            if self.data[i,j] < s:
                leftIndex.append(i)
            else: rightIndex.append(i)
        if len(leftIndex)==0 or len(leftIndex) == len(self.y):
            return
        self.isLeaf = False
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
#计算准确率
def accuracy(T,X,y):
    pred = [prediction(T,X[i,:]) for i in range(X.shape[0])]
    df = pd.DataFrame({'pred':pred,'label':y})
    count = 0
    for i in range(len(y)):
        if pred[i]==y[i]:
            count+=1 
    return count/len(y)

iris = pd.read_csv('./iris.csv')
iris=iris.sample(frac=1).reset_index(drop=True) #打乱数据
iris.head()

catogory = {'setosa':0,'versicolor':1,'virginica':2} #将分类做映射，学习器只能输入数值类型的数据
iris_y = iris['Species'].map(catogory)

iris_dv=iris.drop(columns='Species',axis=1).values #取所有数据要跑很久，所以只取了部分数据

tree = CT(iris_dv,iris_y.values) #建立一颗CART分类树

auc = accuracy(tree,iris_dv,iris_y.values) #算准确率，用训练数据当作测试数据，准确率较高
print("accuracy is {0}".format(auc))
