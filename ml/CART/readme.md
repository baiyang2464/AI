
+ 分类回归决策树

  CART算法采用的是一种二分递归分割的技术，将当前样本分成两个子样本集，使得生成的非叶子节点都有两个分支。因此CART实际上是一颗二叉树。 

  CART可以进行分类也可以用作回归，两者的不同之处在于**如何选择最优的特征及特征的切分点**。用作回归时，使用均方误差来做特征切分评价的好坏；用作分类时，使用基尼系数来做特征切分评价的好坏。

+ 代码

  + 多分类代码

    数据为鸢尾花卉三分类数据集

    [CART_Multi_Classification.ipynb](https://github.com/baiyang2464/AI/blob/master/ml/CART/CART_Multi_Classification.ipynb) 

    [CART_Multi_Classification.py](https://github.com/baiyang2464/AI/blob/master/ml/CART/CART_Multi_Classification.py) 

  + 回归代码

    数据为波士顿房价数据集

    [CART_Regression.ipynb](https://github.com/baiyang2464/AI/blob/master/ml/CART/CART_Regression.ipynb) 

    [CART_Regression.py](https://github.com/baiyang2464/AI/blob/master/ml/CART/CART_Regression.py) 

  + 每个算法的代码择一即可

+ 运行环境


    + ipynb文件:jupyter notebook
    + py文件：python3
    + 工具包：sys、numpy、pandas


+ 数据

  [鸢尾花卉三分类数据集](https://github.com/baiyang2464/AI/blob/master/ml/CART/iris.csv) 

  [波士顿房价数据集](https://github.com/baiyang2464/AI/blob/master/ml/CART/housing.csv) 