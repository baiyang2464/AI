**0.卷积作用的源图片**

<p align="center">
	<img src=./pictures/pic2.jpg alt="Sample"  width="900">
	<p align="center">
		<em> </em>
	</p>
</p>

**1.基于opencv及C++语言实现卷积操作**

+ 环境：win10  vs2017 opencv3.4
+ opencv用于读取jpg文件及保存卷积操作后的数据为图片
  + 安装opencv请看这篇[帖子](<https://www.cnblogs.com/jisongxie/p/9316283.html>)
  + imread与imwrite分别是opencv读取图片和保存图片的函数
+ 用C++实现矩阵的卷积操作

卷积操作及各种不同的卷积核介绍[请看这里](<https://blog.csdn.net/chaipp0607/article/details/72236892?locationNum=9&fps=1>)

**2.二维卷积的简易实现**

![img](https://upload-images.jianshu.io/upload_images/6802183-f80f54f6e05038c4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/277/format/webp)

用4个for循环就可以完成卷积操作

```c++
#include <iostream>
#include <vector>
using std::cout;
using std::endl;
using std::vector;

void conv2d(const vector<vector<int>> &picture,const vector<vector<int>>&kernel,
            vector<vector<int> >&res,int picW,int picH,int kerW,int kerH)
{
	int sum;
	for(int i = 0;i<=(picW-kerW);++i)
		for(int j=0;j<=(picH-kerH);++j)
		{
			sum=0;
			for(int u=0;u<kerW;++u)
				for(int v=0;v<kerH;++v)
				{
					sum+=picture[i+u][j+v]*kernel[u][v];
				}
			res[i][j]=sum;
		}
}

int main()
{
	vector<vector<int> > picture={
		{1,1,1,1,1},
		{-1,0,-3,0,1},
		{2,1,1,-1,0},
		{0,-1,1,2,1},
		{1,2,1,1,1}
	};
	vector<vector<int> > kernel={
		{-1,0,0},
		{0,0,0},
		{0,0,1}
	};
	int picH=5,picW=5;
	int kerH=3,kerW=3;
	int outH = picH-kerH+1;
	int outW = picW-kerW+1;
	vector<vector<int> >res(picW-kerW+1,vector<int>(picH-kerH+1,0));
	conv2d(picture,kernel,res,picH,picW,kerH,kerW);
	for(int u=0;u<outW;++u)
	{
		for(int v=0;v<outH;++v)
		{
			cout<<res[u][v]<<" ";
		}
		cout<<endl;
	}
	return 0;
}

```



**3.将卷积运算作用于图片上**

定义一个my_convolution类，源文件、头文件及main文件：

+ [my_convolution.h](./my_convolution.h)

+ [my_convolution.cpp](./my_convolution.cpp)

+ [main.cpp](./main.cpp)

运行后，经过卷积操作的图片会被保存在`result`文件夹中



**4.代码中尝试了几种卷积核，作下简要介绍**

+ 高斯卷积核：虽然看起来像是模糊了原图片，但作用是用来平滑图片，效果好于均值平滑

<p align="center">
	<img src=https://img-blog.csdn.net/20180701184737562?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoYWlwcDA2MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70 alt="Sample"  width="200">
	<p align="center">
		<em>高斯卷积核</em>
	</p>
</p>
<p align="center">
	<img src=./result/gaussian_smooth.jpg  width="400">
	<p align="center">
		<em>高斯卷积核</em>
	</p>
</p>


+ 图像锐化卷积核

<p align="center">
  	<img src=https://img-blog.csdn.net/20180411211719651?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoYWlwcDA2MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70 alt="Sample"  width="200">
  	<p align="center">
  		<em>锐化卷积核</em>
  	</p>
  </p>
<p align="center">
	<img src=./result/sharpen_9.jpg  width="400">
	<p align="center">
		<em>锐化卷积核</em>
	</p>
</p>


+ sobel边缘检测：有纵向和横向两种

  + 横向
<p align="center">
  	<img src=https://img-blog.csdn.net/2018041121220776?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoYWlwcDA2MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70 alt="Sample"  width="200">
  	<p align="center">
  		<em>横向边缘检查卷积核</em>
  	</p>
  </p>
<p align="center">
	<img src=./result/sobel_y_edge_detecion.jpg  width="400">
	<p align="center">
		<em>横向边缘检查卷积核</em>
	</p>
</p>


  + 纵向

  <p align="center">
  	<img src=https://img-blog.csdn.net/20180411212247770?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoYWlwcDA2MDc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70 alt="Sample"  width="200">
  	<p align="center">
  		<em>纵向边缘检查卷积核</em>
  	</p>
  </p>
<p align="center">
	<img src=./result/sobel_x_edge_detecion.jpg  width="400">
	<p align="center">
		<em>纵向边缘检查卷积核</em>
	</p>
</p>





