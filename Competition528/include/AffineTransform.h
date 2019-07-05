#ifndef   __AFFINETRANSFORM_H__ 
#define   __AFFINETRANSFORM_H__ 

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>

const char* wndname = "Square Detection";
int Max_X = 640, Max_y = 480;//depends on camera
int thresh_Af = 20, N = 11;

using namespace cv;


std::vector<double> GetYawFromFrontCam(std::vector<std::vector<Point> > &squares, float scale)
{
	std::vector<double> Angle;
	Angle.clear();
	int n = squares.size();
	double angletmp = 0, circumference = 0, dis = 2;
	for (size_t i = 0; i < n; i++)
	{
		double x = 0, y = 0;
		for (int j = 0; j<(int)squares[i].size(); j++)
		{
			Point* p = &squares[i][j];
			x += (double)(p->x);
			y += (double)(p->y);
		}
		x /= n;
		y /= n;
		double deltaX = abs(x - Max_X / 2);
		Angle.push_back(atan(deltaX*scale / dis));
	}
	return Angle;
}

static double angle( Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void findSquares(const Mat& image, std::vector<std::vector<Point> >& squares, int mode)
{
	squares.clear();

	Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	// down-scale and upscale the image to filter out the noise
	pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
	pyrUp(pyr, timg, image.size());
	std::vector<std::vector<Point> > contours;

	// find squares in every color plane of the image
	for (int c = 0; c < 3; c++)
	{
		int ch[] = { c, 0 };
		mixChannels(&timg, 1, &gray0, 1, ch, 1);

		// try several threshold levels
		for (int l = 0; l < N; l++)
		{
			// hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading
			if (l == 0)
			{
				// apply Canny. Take the upper threshold from slider
				// and set the lower to 0 (which forces edges merging)
				Canny(gray0, gray, 0, thresh_Af, 3);
				// dilate canny output to remove potential
				// holes between edge segments
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else
			{
				// apply threshold if l!=0:
				//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
				gray = gray0 >= (l + 1) * 255 / N;
			}

			// find contours and store them all as a list
			findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

			std::vector<Point> approx;

			// test each contour
			for (size_t i = 0; i < contours.size(); i++)
			{
				// approximate contour with accuracy proportional
				// to the contour perimeter
				approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

				// square contours should have 4 vertices after approximation
				// relatively large area (to filter out noisy contours)
				// and be convex.
				// Note: absolute value of an area is used because
				// area may be positive or negative - in accordance with the
				// contour orientation
				if (approx.size() == 4 && isContourConvex(Mat(approx)))
				{
					if (mode == 3)
					{
						if (fabs(contourArea(Mat(approx))) > 800)
						{
							double maxCosine = 0;

							for (int j = 2; j < 5; j++)
							{
								// find the maximum cosine of the angle between joint edges
								double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
								maxCosine = MAX(maxCosine, cosine);
							}

							// if cosines of all angles are small
							// (all angles are ~90 degree) then write quandrange
							// vertices to resultant sequence
							if (maxCosine < 0.6)
							{
								double scale = 1.0*((approx[0].x - approx[1].x)*(approx[0].x - approx[1].x) + (approx[0].y - approx[1].y)*(approx[0].y - approx[1].y)) /
									(1.0*(approx[1].x - approx[2].x)*(approx[1].x - approx[2].x) + (approx[1].y - approx[2].y)*(approx[1].y - approx[2].y));
								if (scale<2 && scale>0.5)
									squares.push_back(approx);
							}
						}

					}
					else
					{
						if (fabs(contourArea(Mat(approx))) > 1000)
						{
							double maxCosine = 0;

							for (int j = 2; j < 5; j++)
							{
								// find the maximum cosine of the angle between joint edges
								double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
								maxCosine = MAX(maxCosine, cosine);
							}

							// if cosines of all angles are small
							// (all angles are ~90 degree) then write quandrange
							// vertices to resultant sequence
							if (maxCosine < 0.3)
							{

								double scale = 1.0*((approx[0].x - approx[1].x)*(approx[0].x - approx[1].x) + (approx[0].y - approx[1].y)*(approx[0].y - approx[1].y)) /
									(1.0*(approx[1].x - approx[2].x)*(approx[1].x - approx[2].x) + (approx[1].y - approx[2].y)*(approx[1].y - approx[2].y));
								if (scale<2 && scale>0.5)
									if ((approx[0].x - approx[1].x)*(approx[0].x - approx[1].x) + (approx[0].y - approx[1].y)*(approx[0].y - approx[1].y)>10000)
										squares.push_back(approx);
							}

						}
					}
				}
			}
		}
	}
}

inline void AffineTransform(Mat img, Mat & imgout, int border_value)
{
	int degree;
	std::vector<std::vector<Point> > squares;
	findSquares(img, squares, 1);
	if (squares.size() > 0)
	{
		std::vector<Point> points = squares[squares.size() - 1];
		//std::cout << points;
		float angle;
		for (int i = 1; i < points.size(); i++)
		{
			if (fabs((float)(points[i].y - points[0].y) / (float)(points[i].x - points[0].x)) < 0.5)
				angle = -atan2((float)(points[i].y - points[0].y), (float)(points[0].x - points[i].x));
		}
		if (angle > CV_PI / 2)
			angle = angle - CV_PI;
		if (angle < -CV_PI / 2)
			angle = angle + CV_PI;

		degree = (int)(angle / CV_PI * 180);//warpAffine默认的旋转方向是逆时针，所以加负号表示转化为顺时针
		double a = sin(angle), b = cos(angle);
		int width = img.cols;
		int height = img.rows;
		int width_rotate = int(width * fabs(b) - height * fabs(a));//height * fabs(a) + 
		int height_rotate = int(height * fabs(b) - width * fabs(a));//width * fabs(a) + 
		if (width_rotate <= 20 || height_rotate <= 20)
		{
			width_rotate = 20;
			height_rotate = 20;
		}
		//旋转数组map
		// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
		// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
		float map[6];
		Mat map_matrix = Mat(2, 3, CV_32F, map);
		// 旋转中心
		CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
		CvMat map_matrix2 = map_matrix;
		cv2DRotationMatrix(center, degree, 1.0, &map_matrix2);//计算二维旋转的仿射变换矩阵
		map[2] += (width_rotate - width) / 2;
		map[5] += (height_rotate - height) / 2;
		//Mat img_rotate;
		//对图像做仿射变换
		//CV_WARP_FILL_OUTLIERS - 填充所有输出图像的象素。
		//如果部分象素落在输入图像的边界外，那么它们的值设定为 fillval.
		//CV_WARP_INVERSE_MAP - 指定 map_matrix 是输出图像到输入图像的反变换，
		int chnnel = img.channels();
		if (chnnel == 3)
			warpAffine(img, imgout, map_matrix, Size(width_rotate, height_rotate), 1, 0, Scalar(border_value, border_value, border_value));
		else
			warpAffine(img, imgout, map_matrix, Size(width_rotate, height_rotate), 1, 0, border_value);
		
	}
	else
	{
		std::cout << "no parking plots!" << std::endl;
		imgout = img;
	}
	
}

#endif
