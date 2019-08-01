#ifndef MY_CONVOLUTION
#define MY_CONVOLUTION

#include <opencv2/opencv.hpp>

class My_Convolution {
public:
	My_Convolution();
	~My_Convolution();
	bool load_kernal(cv::Mat kernal);//���ؾ����
	void convolute(const cv::Mat &image, cv::Mat &dst);//�������

private:
	bool kernal_loaded;//�Ƿ��Ѿ����ؾ����
	cv::Mat curr_kernal;//��ǰ�����
	int bios_x, bios_y;//��¼ƫ����

	void compute_sum_of_product(int i, int j, int chan, cv::Mat &complete_image, cv::Mat & dst);//����ÿһ�����ص���ģ�˻�֮��
	void padling(const cv::Mat &image, cv::Mat &dst);//0��������ʹ�õĵȿ���
};

#endif // MY_CONVOLUTION
#pragma once
