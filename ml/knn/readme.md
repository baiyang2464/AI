
+ knn

  knn算法用相似度来对数据进行分类，每次预测时从训练数据中找出k条与用例数据最相似的数据，然后对k条数据采用某种分类决策方法决定用例数据属于哪一类

  相似度计算：可以采用曼哈顿距离、欧式距离或余旋相似度等

  分类决策方法：常见的有投票原则（即多数票原则）

  算法较为简单，可以直接看代码

+ 代码

  [knn.ipynb](https://github.com/baiyang2464/AI/blob/master/ml/knn/knn.ipynb) 

  [knn.py](https://github.com/baiyang2464/AI/blob/master/ml/knn/knn.py) 

  以上代码择一即可

+ 运行环境


    + ipynb文件:jupyter notebook
    + py文件：python3
    + 工具包：sys、numpy、pandas


+ 数据

  [鸢尾花卉三分类数据集](https://github.com/baiyang2464/AI/blob/master/ml/CART/iris.csv) 
