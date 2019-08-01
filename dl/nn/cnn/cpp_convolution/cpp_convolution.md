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

void conv2d(const vector<vector<int> > &picture,const vector<vector<int> > &kernel,vector<vector<int> >&res,int picW,int picH,int kerW,int kerH)
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

定义一个my_convolution类，源文件头文件及main文件

```c++
//my_convolution.h
#ifndef MY_CONVOLUTION
#define MY_CONVOLUTION

#include <opencv2/opencv.hpp>

class My_Convolution {
public:
	My_Convolution();
	~My_Convolution();
	bool load_kernal(cv::Mat kernal);//加载卷积核
	void convolute(const cv::Mat &image, cv::Mat &dst);//卷积操作

private:
	bool kernal_loaded;//是否已经加载卷积核
	cv::Mat curr_kernal;//当前卷积核
	int bios_x, bios_y;//记录偏移量

	void compute_sum_of_product(int i, int j, int chan, cv::Mat &complete_image, cv::Mat & dst);//计算每一个像素的掩模乘积之和
	void padling(const cv::Mat &image, cv::Mat &dst);//0的填充
};

#endif // MY_CONVOLUTION
#pragma once

```

```c++
//my_convolution.cpp
#include "my_convolution.h"

using namespace std;
using namespace cv;

My_Convolution::My_Convolution() {
	kernal_loaded = false;
}
My_Convolution::~My_Convolution() {}

//加载卷积核
bool My_Convolution::load_kernal(Mat kernal) {
	if (kernal.cols % 2 == 1 && kernal.rows % 2 == 1) {
		curr_kernal = kernal.clone();
		bios_x = (kernal.cols - 1) / 2;
		bios_y = (kernal.rows - 1) / 2;
		kernal_loaded = true;
		return true;
	}
	else {
		cout << "The size of kernal is not suitable!" << endl;
		return false;
	}
}

//卷积操作
void My_Convolution::convolute(const Mat &image, Mat &dst) {
	if (!kernal_loaded) {
		cout << "kernal is empty!Please load the kernal first!" << endl; return;
	}
	Mat complete_image;
	padling(image, complete_image);
	dst = Mat::zeros(image.rows, image.cols, image.type());
	int channels = image.channels();//获取图像的通道数
	if (channels == 3) {
		for (int chan = 0; chan < channels; chan++) {
			for (int i = 0; i < dst.rows; i++) {
				for (int j = 0; j < dst.cols; j++) {
					compute_sum_of_product(i, j, chan, complete_image, dst);
				}
			}
		}
		return;
	}
	if (channels == 1) {
		for (int i = 0; i < dst.rows; i++) {
			for (int j = 0; j < dst.cols; j++) {
				compute_sum_of_product(i, j, 0, complete_image, dst);
			}
		}
	}

}

//计算掩模乘积之和
void My_Convolution::compute_sum_of_product(int i, int j, int chan, Mat &complete_image, Mat &dst) {
	if (complete_image.channels() == 3) {
		float sum = 0;
		int bios_rows = i;
		int bios_cols = j;
		for (int curr_rows = 0; curr_rows < curr_kernal.rows; curr_rows++) {
			for (int curr_cols = 0; curr_cols < curr_kernal.cols; curr_cols++) {
				float a = curr_kernal.at<float>(curr_rows, curr_cols)*complete_image.at<Vec3b>(curr_rows + bios_rows, curr_cols + bios_cols)[chan];
				sum += a;
			}
		}
		dst.at<Vec3b>(i, j)[chan] = (int)sum;
	}
	else {
		if (complete_image.channels() == 1) {
			float sum = 0;
			int bios_rows = i;
			int bios_cols = j;
			for (int curr_rows = 0; curr_rows < curr_kernal.rows; curr_rows++) {
				for (int curr_cols = 0; curr_cols < curr_kernal.cols; curr_cols++) {
					float a = curr_kernal.at<float>(curr_rows, curr_cols)*complete_image.at<uchar>(curr_rows + bios_rows, curr_cols + bios_cols);
					sum += a;
				}
			}
			dst.at<uchar>(i, j) = (int)sum;
		}
		else {
			cout << "the type of image is not suitable!" << endl; return;
		}
	}


}

//0填充
void My_Convolution::padling(const Mat &image, Mat &dst) {
	if (!kernal_loaded) {
		cout << "kernal is empty!" << endl;
		return;
	}
	dst = Mat::zeros(2 * bios_y + image.rows, 2 * bios_x + image.cols, image.type());//初始化一个补全图像的大小。
	Rect real_roi_of_image = Rect(bios_x, bios_y, image.cols, image.rows);
	Mat real_mat_of_image = dst(real_roi_of_image);
	image.copyTo(dst(real_roi_of_image));
}

```

```c++
//main.cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "my_convolution.h"

using namespace std;
using namespace cv;

//高斯核构造函数
Mat Gaussian_kernal(int kernal_size, int sigma)
{
	const double PI = 3.14159265358979323846;
	int m = kernal_size / 2;
	Mat kernal(kernal_size, kernal_size, CV_32FC1);
	float s = 2 * sigma*sigma;
	for (int i = 0; i < kernal_size; i++)
	{
		for (int j = 0; j < kernal_size; j++)
		{
			int x = i - m, y = j - m;
			kernal.ptr<float>(i)[j] = exp(-(x*x + y * y) / s) / (PI*s);
		}
	}
	return kernal;
}

//sobel边缘检测算子
cv::Mat sobel_y_kernal = (Mat_<float>(3, 3) << -1, -2, -1,
	0, 0, 0,
	1, 2, 1);
cv::Mat sobel_x_kernal = (Mat_<float>(3, 3) << -1, 0, 1,
	-2, 0, 2,
	-1, 0, 1);

//图像锐化
cv::Mat sharpen_kernal = (Mat_<float>(3, 3) << -1, -1, -1,
	-1, 9, -1,
	-1, -1, -1);

int main() {
	My_Convolution myconvolution;
	Mat image = imread("./pictures/pic2.jpg");
	imshow("src", image);

	//锐化
	Mat dst_sharpen;
	myconvolution.load_kernal(sharpen_kernal);
	myconvolution.convolute(image, dst_sharpen);

	//imshow("dsgaussian", dst_sharpen);
	imwrite("./result/sharpen.jpg", dst_sharpen);
	
	//高斯卷积
	Mat dst_gaussian;
	myconvolution.load_kernal(Gaussian_kernal(7, 2));
	myconvolution.convolute(image, dst_gaussian);

	//imshow("dst_gaussian", dst_gaussian);
	imwrite("./result/gaussian_smooth.jpg", dst_gaussian);

	//sobel_边缘检测操作
	Mat dst_sobel_x;
	Mat dst_sobel_y;

	myconvolution.load_kernal(sobel_x_kernal);
	myconvolution.convolute(image, dst_sobel_x);
	//imshow("dst_sobel_x", dst_sobel_x);
	imwrite("./result/sobel_x_edge_detecion.jpg", dst_sobel_x);

	myconvolution.load_kernal(sobel_y_kernal);
	myconvolution.convolute(image, dst_sobel_y);
	//imshow("dst_sobel_y", dst_sobel_y);
	imwrite("./result/sobel_y_edge_detecion.jpg", dst_sobel_y);
	
	waitKey(0);
	return 0;
}

```

运行后，经过卷积操作的图片会被保存在`result`文件夹中

代码中尝试了几种卷积核，作下简要介绍

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





