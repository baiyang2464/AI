#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd 

#处理数据，将数据打乱，将字符型标签映射成数字
iris = pd.read_csv("./iris.csv")
data=iris.sample(frac=1).reset_index(drop=True) #打乱数据
catogory = {'setosa':0,'versicolor':1,'virginica':2} #将分类做映射，学习器只能输入数值类型的数据
data['Species'] = data['Species'].map(catogory)
data.head()

#knn算法实现
#距离度量——计算欧式距离
def calDistance(veca,vecb):#输入向量必须是np.array一维数组
    return np.sqrt(np.sum(np.square(veca-vecb)))

#分类决策——投票原则
def maxVoter(arr):
    dic = {}
    for item in arr:#统计各类别出现的次数
        if item not in dic:
            dic[item] = 1 
        else:
            dic[item] +=1 
    return max(dic,key=dic.get)#输出类别出现次数最多的类


#用线性搜索法找到与输入用例最相似的k条数据
#用投票的方法得出实例的类别
data['Dist'] = 0 
def knn(k,target,data):
    m,n = data.shape
    for row in range(m):
        curV = data.drop(columns=['Species','Dist'],axis=1).loc[row,:].values
        data.loc[row,'Dist'] = calDistance(target,curV)#计算用例与训练数据集合中样本的距离
    kRows=data.sort_values(by ='Dist',ascending=True).iloc[0:k,:] #按距离进行递增排序，选出距离最近的前5条
    pred = maxVoter(kRows['Species'].values)#投票得出分类结果
    return pred

#将150条数据前100条分成训练数据，后50条分成测试数据
train = data.loc[:100,:]
test = data.loc[100:,:].reset_index(drop=True)

#训练 和 预测
count = 0
catogoryReverse = {0:'setosa',1:'versicolor',2:'virginica'}
for i in range(test.shape[0]):
    pred = knn(5,test.drop(columns=['Species','Dist'],axis=1).iloc[i,:].values,train)
    y = test.loc[i,'Species']
    print("step is %d, predict %s to %s"%(i,catogoryReverse[y],catogoryReverse[pred]))
    if pred !=y:
        count+=1
print("accuracy is %f"%(1-float(count/test.shape[0])))

