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

//0填充操作
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
