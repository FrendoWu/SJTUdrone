#ifndef IMAGE_CONSTRUCTION
#define IMAGE_CONSTRUCTION

#include <opencv2/core/core.hpp>    
#include <opencv2/highgui/highgui.hpp>   
#include "imgproc/imgproc.hpp"    
#include <opencv2/features2d/features2d.hpp>   
#include "opencv2/xfeatures2d/nonfree.hpp"   
#include <vector>  
#include "opencv2/opencv.hpp"  
#include "detect_circle.h"
#include "detect_num_nav.h"
#include "opencv2/imgproc/imgproc.hpp"


using namespace cv;
using namespace std;
//计算原始图像点位在经过矩阵变换后在目标图像上对应位置    
Point2f getTransformPoint(const Point2f originalPoint, const Mat &transformMaxtri)
{
	Mat originelP, targetP;
	originelP = (Mat_<double>(3, 1) << originalPoint.x, originalPoint.y, 1.0);
	targetP = transformMaxtri * originelP;
	float x = targetP.at<double>(0, 0) / targetP.at<double>(2, 0);
	float y = targetP.at<double>(1, 0) / targetP.at<double>(2, 0);
	return Point2f(x, y);
}

cv::Mat image_construction(std::vector<cv::Mat> all_image)
{
	std::vector<cv::Mat>::iterator it = all_image.begin();
	cv::Mat image02 = *it;
	for (int y = 0; y < image02.rows; y++)
		for (int x = 0; x < image02.cols; x++)
			for (int c = 0; c < 3; c++)
				image02.at<Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(1.4*image02.at<Vec3b>(y, x)[c] + 10);
	cv::Mat image01;
	std::string filename = "D://intermidate";
	int count = 1;

	while (1)
	{
		it++;
	    image01 = *it;
		for (int y = 0; y < image01.rows; y++)
			for (int x = 0; x < image01.cols; x++)
				for (int c = 0; c < 3; c++)
					image01.at<Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(1.4*image01.at<Vec3b>(y, x)[c] + 10);
		//灰度图转换    
		Mat image1, image2;
		cv::cvtColor(image01, image1, CV_RGB2GRAY);
		cv::cvtColor(image02, image2, CV_RGB2GRAY);
		//提取特征点      
		int minHessian = 400;
		Ptr<xfeatures2d::SURF> suftDetector = xfeatures2d::SURF::create(minHessian);
		std::vector<KeyPoint> keyPoint1, keyPoint2;
		suftDetector->detect(image1, keyPoint1);
		suftDetector->detect(image2, keyPoint2);
		//特征点描述，为下边的特征点匹配做准备      
		Mat imageDesc1, imageDesc2;
		suftDetector->compute(image1, keyPoint1, imageDesc1);
		suftDetector->compute(image2, keyPoint2, imageDesc2);
		//获得匹配特征点，并提取最优配对       
		FlannBasedMatcher matcher;
		std::vector<DMatch> matchePoints;
		matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
		sort(matchePoints.begin(), matchePoints.end()); //特征点排序      
											//获取排在前N个的最优匹配特征点    
		std::vector<Point2f> imagePoints1, imagePoints2;
		for (int i = 0; i < 10; i++)
		{
			imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);
			imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);
		}

		//获取图像1到图像2的投影映射矩阵，尺寸为3*3
		Mat adjustMat = (Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);

		//获取最强配对点在原始图像和矩阵变换后图像上的对应位置，用于图像拼接点的定位    
		Point2f originalLinkPoint, targetLinkPoint, basedImagePoint;
		originalLinkPoint = keyPoint1[matchePoints[0].queryIdx].pt;
		targetLinkPoint = getTransformPoint(originalLinkPoint, adjustMat);
		basedImagePoint = keyPoint2[matchePoints[0].trainIdx].pt;
	
		//图像配准    
		Mat imageTransform1;
		warpPerspective(image01, imageTransform1, adjustMat, Size(image02.cols, image02.rows + image01.rows - basedImagePoint.y ));
		cv::imwrite("persp.jpg", imageTransform1);
		//在最强匹配点的位置处衔接，最强匹配点左侧是图1，右侧是图2，这样直接替换图像衔接不好，光线有突变   
		Mat ROIMat = image02(Rect(Point(0, basedImagePoint.y + 480 - targetLinkPoint.y), Point(image02.cols, image02.rows)));
		ROIMat.copyTo(Mat(imageTransform1, Rect(0, 480, image02.cols, image02.rows - basedImagePoint.y + targetLinkPoint.y - 480 + 1)));
	
		image02 = imageTransform1;
		if ((it+1) == all_image.end())
		{
			return imageTransform1;
		}
		cv::imwrite(filename + std::to_string(count++) + std::string(".jpg"), image02);
	}
}

bool comp(const Vec3i &a, const Vec3i &b)
{
	return a[0] < b[0];
}

std::vector<Vec3f> calculate(float height, std::vector<cv::Mat> all_image, std::vector<cv::Mat> all_image2)
{
	cv::Mat image;
	Mat st;
	cv::Mat image2;
	std::vector<cv::Mat> ImageForPosition;
	image = image_construction(all_image);
	//ImageForPosition.push_back(image);
	//image2 = image_construction2(all_image2);
	//ImageForPosition.push_back(image2);
	//image = image_construction(ImageForPosition);

	cv::imwrite("D:\\aresult.jpg", image);
	st = Mat::zeros(image.size(), image.type());
	image.convertTo(st, -1, 0.5, 0);
	cv::imwrite("D:\\aresult_downlight.jpg", st);

	std::vector<cv::Mat> all_images;
	std::vector<Vec4i> range;
	std::vector<Vec3i> circles = detect_circle(image, range);
	std::vector<Vec3i> nums = detect_num_nav(image, all_images);
	std::vector<Vec3i> combine;
	std::vector<Vec3f> distance;
	Vec3f dist;
	combine.insert(combine.end(), circles.begin(), circles.end());
	combine.insert(combine.end(), nums.begin(), nums.end());
	sort(combine.begin(), combine.end(), comp);
	std::reverse(combine.begin(), combine.end());
	std::vector<Vec3i>::iterator circles_it = circles.begin();
	std::vector<Vec3i>::iterator nums_it = nums.begin();
	std::vector<Vec3i>::iterator combine_it = combine.begin() + 1;
	for (; combine_it != combine.end() - 1; combine_it++)
	{
		if (abs((*combine_it)[0] - (*(combine_it + 1))[0]) < 50)
		{
			(*combine_it)[2] = 0;
			(*(combine_it + 1))[0] = (*combine_it)[0];
			(*(combine_it + 1))[1] = (*combine_it)[1];
			(*(combine_it + 1))[2] = 0;
			continue;
		}
		dist[0] = ((*combine_it)[0] - (*(combine_it + 1))[0]) * height / 269.5;
		dist[1] = ((*combine_it)[1] - (*(combine_it + 1))[1]) * height / 269.5;
		dist[2] = (float)(*(combine_it + 1))[2];
		distance.push_back(dist);
	}
	dist[0] = ((*combine.begin())[0] - (*(combine_it + 1))[0]) * height / 269.5;
	dist[1] = ((*combine.begin())[1] - (*(combine_it + 1))[1]) * height / 269.5;//距离计算，需要修改
	dist[2] = (float)(*(combine_it + 1))[2];
	distance.push_back(dist);



	return distance;
	
}

#endif // !IMAGE_CONSTRUCTION
