
+ 主成分分析

  在主成分分析中，它试图找到一个低维的空间，来对数据进行投影，以便最小化投影误差的平方，即最小化每个点与投影后相对应的点距离的平方，实际是求最小投射平方误差。  

  PCA变换其实就是一种降维技术。上文中对原始数据**选择投影面**可以用**矩阵变换来**做。 

  涉及到协方差矩阵、特征向量、特征值的知识

  [放一篇讲解到位的博客](https://blog.csdn.net/a10767891/article/details/80288463)

+ 代码

  [PCA.ipynb](https://github.com/baiyang2464/AI/blob/master/ml/PCA/PCA.ipynb) 

  [PCA.py](https://github.com/baiyang2464/AI/blob/master/ml/PCA/PCA.py) 

  以上两个算法的代码择一即可

+ 运行环境


    + ipynb文件:jupyter notebook
    + py文件：python3
    + 工具包：sys、numpy、pandas


+ 数据

  sklearn的鸢尾花卉三分类数据集

  将其特征从四位降到了二维，并可视化的展示了出来