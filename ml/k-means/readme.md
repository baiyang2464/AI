
+ k-means

  聚类假设数据具有类别特征，可以依类别特征来区分数据，使得类内之间的数据最为相似，各类之间的数据相似度差别尽可能大。 

   k-means算法是一种简单的迭代型聚类算法，采用欧式距离作为相似性指标，从而发现给定数据集中的K个类，且每个类的中心是根据类中所有值的均值得到，每个类用聚类中心来描述。聚类目标是使得各类的聚类平方和最小，即最小化： 

  ![](https://images0.cnblogs.com/blog2015/771535/201508/071351008301642.jpg)

  同时为了使得算法收敛，在迭代过程中，应使最终的聚类中心尽可能的不变。 

+ 算法步骤

  K-means是一个反复迭代的过程，算法分为四个步骤：

  1） 选取数据空间中的K个对象作为初始中心，每个对象代表一个聚类中心；

  2） 对于样本中的数据对象，根据它们与这些聚类中心的欧氏距离，按距离最近的准则将它们分到距离它们最近的聚类中心（最相似）所对应的类；

  3） 更新聚类中心：将每个类别中所有对象所对应的均值作为该类别的聚类中心，计算目标函数的值；

  4） 判断聚类中心是否发生改变，若不变，则输出结果，若改变，则返回2）。

+ 代码

  + 脚本

    [k-means.py](https://github.com/baiyang2464/AI/blob/master/ml/k-means/k-means.py) 

  + jupyter notebook

    [k-means.ipynb](https://github.com/baiyang2464/AI/blob/master/ml/k-means/k-means.ipynb) 

  + 每个算法的代码择一即可

+ 运行环境


    + ipynb文件:jupyter notebook
    + py文件：python3
    + 工具包：matplotlib.pyplot、numpy、pandas


+ 数据

  [自己做的多分类数据](https://github.com/baiyang2464/AI/blob/master/ml/k-means/data.csv) 
