#ifndef DETECT_CIRCLE_H
#define DETECT_CIRCLE_H

#include <iostream>  
#include <string>  
#include <list>  
#include <vector>  
#include <map>  
#include <stack>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include "detect_num.h"

using namespace std;
using namespace cv;

//调用此函数的方法
//cv::Mat origin_image;
//vector<cv::Mat> mats;
//vector<Vec2i> vec = detect_num(origin_image,&mats);(返回每个停机坪的中心点坐标，vec2i包含两个整数)
//cout << vec.begin()[0][0];(第一个停机坪中心点的x)

double Circle_CalculateTheta(Vec2d &oc, Vec2d &pt_0, Vec2d &pt_x);
bool Circle_PointCmp(Vec2d &a, Vec2d &b, Vec2d &center);
void Circle_ClockSortPoints(std::vector<Vec2d> &vPoints, Vec2d &center, Vec2d &pt_0);

void Circle_FindMinMax(int &x_min, int &x_max, int &y_min, int &y_max, Mat image)
{
	x_max = 0;
	y_max = 0;
	x_min = image.rows;
	y_min = image.cols;

}

static void Circle_ImageThreshold(Mat imgOriginal, Mat &imgThresholded)
{

	int iLowH = 0;
	int iHighH = 10;

	int iLowS = 43;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;


	bool found = false;

	Mat imgHSV;
	std::vector<Mat> hsvSplit;
	cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
	split(imgHSV, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, imgHSV);

	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);


	morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
	Mat img_canny;
	Canny(imgThresholded, img_canny, 10, 80);
	std::vector<std::vector<Point> > contours, good_contours;
	findContours(img_canny, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);


	std::vector<Point2f> imagePoints_vertex;
	std::vector<Point> approxPoly;
	if (contours.size() == 0)
	{
		std::cout << "no contour is found" << endl;
	}
	else
	{
		for (int k = 0; k < contours.size(); k++)
		{
			if (contourArea(contours[k]) > 10000)
				approxPolyDP(contours[k], approxPoly, 1, 1);
			if (approxPoly.size() == 4)
				;
		}
	}

	imwrite("ThresholdedCircle.jpg", imgThresholded);

	char key = (char)waitKey(30);
	if (key == 27)
		std::cout << 0;
}

inline double Circle_CalculateTheta(Vec2d &oc, Vec2d &pt_0, Vec2d &pt_x)
{
	double line1_length_2 = sqrt(((pt_0[0] - oc[0])*(pt_0[0] - oc[0]) + (pt_0[1] - oc[1])*(pt_0[1] - oc[1])));
	double line2_length_2 = sqrt(((pt_x[0] - oc[0])*(pt_x[0] - oc[0]) + (pt_x[1] - oc[1])*(pt_x[1] - oc[1])));
	double product = (pt_0[0] - oc[0])*(pt_x[0] - oc[0]) + (pt_0[1] - oc[1])*(pt_x[1] - oc[1]);
	double cos_theta = product / (line1_length_2*line2_length_2);
	if (Circle_PointCmp(pt_0, pt_x, oc))
		return 6.28 - acos(cos_theta);
	else
		return acos(cos_theta);
}

inline bool Circle_PointCmp(Vec2d &a, Vec2d &b, Vec2d &center)
{
	int det = (a[0] - center[0]) * (b[1] - center[1]) - (b[0] - center[0]) * (a[1] - center[1]);
	int mult = (a[0] - center[0]) *(b[0] - center[0]) + (a[1] - center[1]) * (b[1] - center[1]);

	if (det < 0)
		return true;
	if (det > 0)
		return false;

	if (det == 0)
	{
		if (mult>0) return false;
		else return true;
	}
}

inline  void Circle_ClockSortPoints(std::vector<Vec2d> &vPoints, Vec2d &center, Vec2d &pt_0)
{
	for (int i = 0; i < vPoints.size() - 1; i++)
	{
		for (int j = 0; j < vPoints.size() - i - 1; j++)
		{
			if (CalculateTheta(center, pt_0, vPoints[j])>CalculateTheta(center, pt_0, vPoints[j + 1]))
			{
				Vec2d tmp = vPoints[j];
				vPoints[j] = vPoints[j + 1];
				vPoints[j + 1] = tmp;
			}
		}
	}
}

void  Circle_Two_PassNew(const Mat &bwImg, Mat &labImg)
{
	assert(bwImg.type() == CV_8UC1);
	labImg.create(bwImg.size(), CV_32SC1);
	labImg = Scalar(0);
	labImg.setTo(Scalar(1), bwImg);
	assert(labImg.isContinuous());
	const int Rows = bwImg.rows - 1, Cols = bwImg.cols - 1;
	int label = 1;
	std::vector<int> labelSet;
	labelSet.push_back(0);
	labelSet.push_back(1);
	//the first pass  
	int *data_prev = (int*)labImg.data;
	int *data_cur = (int*)(labImg.data + labImg.step);
	for (int i = 1; i < Rows; i++)
	{
		data_cur++;
		data_prev++;
		for (int j = 1; j<Cols; j++, data_cur++, data_prev++)
		{
			if (*data_cur != 1)
				continue;
			int left = *(data_cur - 1);
			int up = *data_prev;
			int neighborLabels[2];
			int cnt = 0;
			if (left>1)
				neighborLabels[cnt++] = left;
			if (up > 1)
				neighborLabels[cnt++] = up;
			if (!cnt)
			{
				labelSet.push_back(++label);
				labelSet[label] = label;
				*data_cur = label;
				continue;
			}
			int smallestLabel = neighborLabels[0];
			if (cnt == 2 && neighborLabels[1]<smallestLabel)
				smallestLabel = neighborLabels[1];
			*data_cur = smallestLabel;
			
			for (int k = 0; k<cnt; k++)
			{
				int tempLabel = neighborLabels[k];
				int& oldSmallestLabel = labelSet[tempLabel];
				if (oldSmallestLabel > smallestLabel)
				{
					labelSet[oldSmallestLabel] = smallestLabel;
					oldSmallestLabel = smallestLabel;
				}
				else if (oldSmallestLabel<smallestLabel)
					labelSet[smallestLabel] = oldSmallestLabel;
			}
		}
		data_cur++;
		data_prev++;
	}

	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int prelabel = labelSet[curLabel];
		while (prelabel != curLabel)
		{
			curLabel = prelabel;
			prelabel = labelSet[prelabel];
		}
		labelSet[i] = curLabel;
	}
	//second pass  
	data_cur = (int*)labImg.data;
	for (int i = 0; i < Rows; i++)
	{
		for (int j = 0; j < bwImg.cols - 1; j++, data_cur++)
			*data_cur = labelSet[*data_cur];
		data_cur++;
	}
}
cv::Scalar Circle_GetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}

std::vector<Vec3i> detect_circle(cv::Mat oriImage, std::vector<Vec4i> &range)
{
	std::vector<Vec3i> coordination;
	cv::Mat one_image;
	cv::Mat binImage;
	Circle_ImageThreshold(oriImage, binImage);
	cv::threshold(binImage, binImage, 50, 1, CV_THRESH_BINARY);
	cv::Mat labelImg;
	double time;
	time = getTickCount();
	
	Circle_Two_PassNew(binImage, labelImg);
	
	int *data_cur = (int*)labelImg.data;
	int t1;
	int t2 = 0;
	int x_min = labelImg.rows;
	int y_min = labelImg.cols;
	int x_max = 0;
	int y_max = 0;
	Vec3i zuobiao;
	Vec4i fanwei;
	fanwei[0] = 0;
	fanwei[1] = 0;
	fanwei[2] = 0;
	fanwei[3] = 0;
	while (1)
	{
		int t = 0;
		for (int i = 0; i < labelImg.rows; i++)
		{
			for (int j = 0; j < labelImg.cols - 1; j++, data_cur++)
			{
				if (*data_cur == 0)
					continue;
				t1 = *data_cur;
				if ((t1 != t2) && (t2 > 0))
					continue;
				if (i < x_min)
					x_min = i;
				if (i > x_max)
					x_max = i;
				if (j < y_min)
					y_min = j;
				if (j > y_max)
					y_max = j;
				t2 = t1;
				*data_cur = 0;
				t = 1;
			}
			data_cur++;
		}
		if (!t)
			break;
		if (((x_max - x_min) < 5 && (y_max - y_min) < 5) || (x_min - fanwei[1]) < 100)
		{
			x_min = labelImg.rows;
			y_min = labelImg.cols;
			x_max = 0;
			y_max = 0;
			t2 = 0;
			data_cur = (int*)labelImg.data;
			continue;
		}
		
		zuobiao[0] = (x_max + x_min) / 2;
		zuobiao[1] = (y_min + y_max) / 2;
		zuobiao[2] = 0;
		fanwei[0] = x_min;
		fanwei[1] = x_max;
		fanwei[2] = y_min;
		fanwei[3] = y_max;
		coordination.push_back(zuobiao);
		range.push_back(fanwei);

		x_min = labelImg.rows;
		y_min = labelImg.cols;
		x_max = 0;
		y_max = 0;
		t2 = 0;
		data_cur = (int*)labelImg.data;
	}
	
	return coordination;
}

#endif
