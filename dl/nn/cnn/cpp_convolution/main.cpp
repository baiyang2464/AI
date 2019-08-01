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

