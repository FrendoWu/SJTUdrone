#include "common/common_utils/StrictMode.hpp"
STRICT_MODE_OFF
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif // !RPCLIB_MSGPACK
#include "rpc/rpc_error.h"
STRICT_MODE_ON

#include <Windows.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>

#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include "common/common_utils/FileSystem.hpp"
#include "controllers/PidController.hpp"
#include "sensors/gps/GpsBase.hpp"
#include "common/EarthUtils.hpp"

#include "image_construction.h"
#include "AffineTransform.h"
#include "detect_num_nav.h"
#include "kalman_filter.h"
#include "detect_circle.h"
#include "ImageToMat.h"
#include "detect_num.h"
#include "Ellipse.h"
#include "detect_tree.h"

//相机内参
const double camera_cx = 319.5;
const double camera_cy = 239.5; 
const double camera_fx = 269.5; 
const double camera_fy = 269.5;

//像素点到相机中心的实际误差
Vec2i projectPixelToReal(float alt, Vec2i& target)
{
	float x, y;
	Vec2i real_xy;
	x = alt*(target[0] - camera_cx) / camera_fx;
	y = alt*(target[1] - camera_cy) / camera_fy;
	real_xy[0] = x;
	real_xy[1] = y;
	return real_xy;
}
Mat convertTo3Channels(const Mat& binImg)
{
	Mat three_channel = Mat::zeros(binImg.rows, binImg.cols, CV_32FC3);
	std::vector<Mat> channels;
	for (int i = 0; i < 3; i++)
	{
		channels.push_back(binImg);
	}
	merge(channels, three_channel);
	return three_channel;
}

Ptr<cv::ml::ANN_MLP> bp = cv::ml::ANN_MLP::create();

const float pi = 3.1415926;
float Q = 0.2;   //the process noise variance, used as a parameter for tuning the intensity of the filter action
float Q1 = 0.2;
const float R = 0.2;  //the sensor noise variance
const float R1 = 0.0000047;  //the sensor noise variance--经度
const float R2 = 0.000004;  //the sensor noise variance--纬度
#define CLIP3(_n1, _n,  _n2) {if (_n<_n1) _n=_n1;  if (_n>_n2) _n=_n2;}

void SaveResult(cv::Mat & img, ImageCaptureBase::ImageResponse & response, std::vector< Point2f > corner, int ArucoID)
{
	char savepathpng[100], savepathpmf[100], savepathtxt[100];
	ofstream resultID;
	cv::Mat TempImg;
	cv::Mat imgCodecopy = img.clone();			//保存原图片
	int width = corner.at(1).x - corner.at(0).x;
	int height = corner.at(3).y - corner.at(0).y;
	cv::Mat imageROI = imgCodecopy(Rect(corner.at(0).x, corner.at(0).y, width, height));	
	imageROI.convertTo(TempImg, TempImg.type());		
	sprintf(savepathpng, "D:\\images\\%d.png", ArucoID);
	sprintf(savepathpmf, "D:\\depth\\%d.pfm", ArucoID);
	cv::imwrite(savepathpng, imgCodecopy);
	resultID.open("D:\\result.txt",ios::app);
	resultID << ArucoID << " " << corner.at(0).x << " " << corner.at(0).y
		<< " " << corner.at(2).x << " " << corner.at(2).y << std::endl;
	resultID.close();
	Utils::writePfmFile(response.image_data_float.data(), response.width, response.height,
		savepathpmf);
}
int main()
{
	using namespace std;
	using namespace msr::airlib;

	cv::Mat img, image_depth, image_depth_C1;

	msr::airlib::MultirotorRpcLibClient client;
	ImageToMat img2mat;
	typedef ImageCaptureBase::ImageRequest ImageRequest;
	typedef ImageCaptureBase::ImageResponse ImageResponse;
	typedef ImageCaptureBase::ImageType ImageType;
	typedef common_utils::FileSystem FileSystem;
 
	try
	{
		client.confirmConnection();
		client.enableApiControl(true);
		client.armDisarm(true);
		BarometerData curr_bardata;
		ImuData Imudata;
		MagnetometerData Target_Magdata, Magdata;
		Target_Magdata = client.getMagnetometerdata();
		float target_Mag_y = -Target_Magdata.magnetic_field_body.y();
		Vector3r ned_origin, ned_curr, ned_target, home_ned, control_origin;
		Vector3r ArucoBegin;
		BarometerData Barometer_origin = client.getBarometerdata();
		BarometerData point_control_bardata;

		int count0 = 0, count1 = 0, count2 = 0, count3 = 0, count4 = 0, count_circle = 0, count_parking = 0, count_parking_1 = 0, count_left = 0, count_right = 0;
		int count_parking_2 = 0;
		int count_parking_local = 0;
		int count_home = 0;
		int count_code = 0, count_yaw = 0;
		int count_go = 0;
		int count_code_alt = 0;
		int currentnumber = 0, nextnumber = 1;
		int TREE_NUM = 0;
		int UPDOWN_COUNT = 0;
		int controlmode = 4;
		int i_kalman = 0; //for kalman z
		int x_kalman = 0;
		int y_kalman = 0;
		int TREE_COUNT = 0;
		int HOME_COUNT = 0;
		int flag_con;
		int flag_dis;
		float test_delta_pitch;
		float theta, target_theta;
		bool flag = false; //for circle
		bool flag_parking = false;
		bool parking = false;
		bool flag_image = false;
		int ten_position = 0;

		cv::Mat image, Img, img4;
		Vec2i xy, xy_temp, XY_TREE;
		Vec2f radius;
		prev_states prev[1] = { 0,0 };
		prev_states prev_x[1] = { 0,0 };
		prev_states prev_y[1] = { 0,0 };
		bool flag_mode3 = false;
		bool isFirst = false;
		bool ReadTxt = false;
		bool FLAG_CB = false;
		bool turnforward = false;
		char traindataPath[256];
		//////// target position ////////////////	
		ned_target(2) = 5.8;//7 下一目标为停机坪  3.3障碍圈
		ned_target(0) = 240;//这里认为图像的纵向为x轴，为了与无人机保持一致
		ned_target(1) = 320;
		//////////pidcontroller////////////////////
		PidController pidX, pidY, pidZ, pidP_X, pidP_Y, pidP_Z, pid_yaw;
		pidP_X.setPoint(ned_target(0), 0.0015, 0, 0.0005);// 这里的x指的是以无人机运动方向为x的反方向
		pidP_Y.setPoint(ned_target(1), 0.0013, 0, 0.0005);
		pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
		pidP_Z.setPoint(ned_target(0), 0.05, 0, 0.0001);
		pid_yaw.setPoint(target_Mag_y, 2, 0, 1);
		flag_con = 0;
		flag_dis = 0;
		/************************** for collect data  ******************************/
		char filename[100];
		char dataPath[256];

		bool flag_collect_data = false;
		bool circle_middle_flag = false, circle_far7_flag = false, circle_far10_flag = false,
			circle_far12_flag = false, circle_far13_flag = false, circle_far16_flag = false, circle_far18_flag = false, circle_far14_flag = false;
		bool last_circle = false;
		bool flag_parking_10 = false;
		bool flag_num10 = false;
		bool FLAG_UPDOWN = true;
		bool is_FLAG_UP = false;
		bool is_FLAG_DOWN = false;
		bool HOME = false;

		int count_num10 = 0;
		int count_collect_data = 0;
		float last_circle_weight = 1;
		float Min_distance = 0;
		std::vector<cv::Mat> ImageForPosition;
		std::vector<cv::Mat> ImageForPosition2;//返回时的
		std::vector<cv::Mat> ImageForPosition3;//合并结果
		std::vector<Vec3f> LocalPosition;
		std::vector<Vec3f> LocalPosition2;
		std::vector<Vec3f> LocalPosition3;

		//////ARuco
		int iter_num;
		int rollCoefficient, pitchCoefficient;
		bool statechange = false;
		bool flag_yaw = false;
		int ArucoID[5];
		int result[5] = { 1,1,1,1,1 };
		int countResult = 0;

		Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);
		Mat out;
		dictionary->drawMarker(100, 600, out, 5);
		Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
		detectorParams->adaptiveThreshWinSizeMin = 3;
		detectorParams->adaptiveThreshWinSizeMax = 3;
		detectorParams->adaptiveThreshWinSizeStep = 1;
		detectorParams->minDistanceToBorder = 0;
		detectorParams->polygonalApproxAccuracyRate = 0.15;
		detectorParams->minMarkerPerimeterRate = 0.15;
		typedef enum enumType
		{
			right2left,
			forward,
			left2right,
		};
		enumType state = right2left;

		while (1)
		{
			clock_t begin = clock();
			if (controlmode == 0)  //0起飞
			{
				//std::cout << " LocalPositionx: " << LocalPosition.at(nextnumber - 1)[0] << " LocalPositiony: " << LocalPosition.at(nextnumber - 1)[1] << std::endl;
				//std::cout << " LocalPosition3: " << LocalPosition.at(nextnumber - 1)[2] << std::endl;
				//curr_bardata = client.getBarometerdata();
				//ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
				//ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
				//std::cout << "ned_curr: " << ned_curr(2) << std::endl;
				//float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
				//std::cout << "delta_throttle: " << delta_throttle << std::endl;
				//CLIP3(0.4, delta_throttle, 0.8);
				//client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);
			//	std::this_thread::sleep_for(std::chrono::duration<double>(0.01f));
			//	std::cout << "count0: " << count0 << std::endl;
				//if (abs(ned_target(2) - ned_curr(2)) < 0.3) count0++;
				//if (count0 > 70)   //需要加判断，下一目标是停机坪或者障碍圈
				//{
				//	if (nextnumber == 2 || nextnumber == 3 || nextnumber == 10)
				//	{
				//		controlmode = 1;
				//		count0 = 0;
				//		flag_image = false;
				//		std::cout << "go for parking....." << std::endl;
				//	}
				//	else if (0.5 < LocalPosition.at(nextnumber - 1)[2]) //停机坪
				//	{
				//		controlmode = 1;
				//		count0 = 0;
				//		std::cout << "go for parking....." << std::endl;
				//	}
				//	else if (LocalPosition.at(nextnumber - 1)[2] < 0.5)   //障碍圈
				//	{
				//		ned_target(2) = 3;
				//		pidX.setPoint(LocalPosition.at(nextnumber - 1)[1], 0.001, 0, 0.002);
				//		pidY.setPoint(LocalPosition.at(nextnumber - 1)[0], 0.001, 0, 0.002);
				//		pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
				//		if (abs(ned_target(2) - ned_curr(2)) < 0.2) count_circle++;
				//		std::cout << "count_circle: " << count_circle << std::endl;
				//		if (count_circle > 25 && (abs(ned_target(2) - ned_curr(2)) < 0.15) )
				//		{
				//			count_circle = 0;
				//			controlmode = 3;
				//			count0 = 0;
				//			std::cout << "go for circle....." << std::endl;
				//		}
				//	}
				//	else
				//	{
				//		curr_bardata = client.getBarometerdata();
				//		ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
				//		ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
				//		ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
				//		std::cout << "ned_curr: " << ned_curr(2) << std::endl;
				//		float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
				//		std::cout << "delta_throttle: " << delta_throttle << std::endl;
				//		CLIP3(0.4, delta_throttle, 0.8);
				//		client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);
				//		std::this_thread::sleep_for(std::chrono::duration<double>(0.01f));
				//	}
				//}
				/*else
				{
					std::cout << "get position failed!!!!!" << std::endl;
				}*/
			}
			else if (controlmode == 1) //下一目标为停机坪
			{				
				image = img2mat.get_below_mat();
				if (image.empty())
				{
					std::cout << "Can not load image... " << std::endl;
				}
				else
				{					
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					std::vector<cv::Mat> mats;
					std::vector<Vec2i> vec = detect_num(image, mats, ned_curr(2));////////////检测停机坪	
					std::cout << " mats.size(): " << mats.size() << std::endl;
					if (vec.size() > 0 || LocalPosition.at(nextnumber - 1)[0]>8 || abs(LocalPosition.at(nextnumber - 1)[0])>8)
					{
						int size = vec.size();
						std::cout << "there are " << size << " parking boards." << std::endl;
						int *number = new int[size];
						cv::Mat image00;
						cv::Mat test_temp;
						for (int i = 0; i < size; i++)
						{
							image00 = mats[i].clone();
							//AffineTransform(mats[i], image00, 255);							
							bp = cv::Algorithm::load<cv::ml::ANN_MLP>("numModel.xml");
							Mat destImg; 
							cvtColor(image00, destImg, CV_BGR2GRAY); // 转为灰度图像
							resize(destImg, test_temp, Size(32, 30), (0, 0), (0, 0), CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
							threshold(test_temp, test_temp, 80, 255, CV_THRESH_BINARY);
							Mat_<float>sampleMat(1, 32*30);
							for (int i = 0; i<32*30; ++i)
							{
								sampleMat.at<float>(0, i) = (float)test_temp.at<uchar>(i / 32, i % 32);
							}

							Mat responseMat;
							bp->predict(sampleMat, responseMat);
							Point maxLoc;
							double maxVal = 0;
							minMaxLoc(responseMat, NULL, &maxVal, NULL, &maxLoc);
							cout << "识别结果：" << maxLoc.x << "	相似度:" << maxVal * 100 << "%" << endl;

							number[i] = maxLoc.x;
							//number[i] = recognize("LeNet-weights", image00);

							imshow("test", image00);
							imwrite("result.jpg", image00);
							cv::waitKey(1);
							std::cout << "number[i]: " << number[i] << std::endl;
							if (number[i] == nextnumber && (maxVal * 100)>65)
							{
								xy_temp = vec.begin()[i];
								currentnumber = nextnumber;
								flag_parking = true;
								parking = false;
								sprintf(traindataPath, "D:\\data\\%d.png", nextnumber);
								imwrite(traindataPath, image00); 
								break;
							}
							if(flag_parking)
							{
								count_parking_1++;
								if (count_parking_1 > 15 )
								{
									if (xy_temp[1] <= 320)
									{
										std::cout << "111111111111111111111" << std::endl;
										client.moveByAngleThrottle(0, 0.01, 0.5899, 0.0f, 0.1f);
										std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
										parking = true;
										count_parking_2++;
									}
									else
									{
										std::cout << "parking................" << std::endl;
										client.moveByAngleThrottle(0, -0.01, 0.5899, 0.0f, 0.1f);
										std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
										parking = true;
										count_parking_2++;
									}
									if (count_parking_2 > 60)
									{
										std::cout << "?????????????????" << std::endl;
										xy_temp = vec.begin()[0];

									}
								}
							}
						}
						std::cout << "current number is: " << currentnumber << "  nextnumber: " << nextnumber << std::endl;
						if (xy_temp[0] == 0 && xy_temp[1] == 0)
						{
							std::cout << "can not find the target number!!!!!" << std::endl;
							if (nextnumber == 10 && LocalPosition.at(nextnumber - 2)[2] == 1)
							{
								if (!flag_num10)
								{
									client.moveByAngleThrottle(-0.10, 0, 0.59, 0.0f, 2.0f);
									std::this_thread::sleep_for(std::chrono::duration<double>(2.0f));
								}
								else
								{
									if (count_num10 < 12)
									{
										client.moveByAngleThrottle(0.0, 0.05, 0.6, 0.0f, 0.1f);
										std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
										count_num10++;
									}
									else if (count_num10 > 12 && count_num10 < 36)
									{
										client.moveByAngleThrottle(0.0, -0.05, 0.6, 0.0f, 0.1f);
										std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
										count_num10++;
									}
									else
									{
										client.moveByAngleThrottle(-0.05, 0, 0.59, 0.0f, 0.05f);
										std::this_thread::sleep_for(std::chrono::duration<double>(0.05f));
										count_num10 = 0;
									}
								}
							}
							else if ((LocalPosition.at(nextnumber - 1)[0] < 8) && abs(LocalPosition.at(nextnumber - 1)[1]) < 5)
							{
								std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
								//float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.1, 0, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}

							else if ((LocalPosition.at(nextnumber - 1)[0] < 8) && abs(LocalPosition.at(nextnumber - 1)[1]) > 5)
							{
								std::cout << "###########################" << std::endl;
								float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.04, 0, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
								client.moveByAngleThrottle(0, 0.08*coff, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}

							else if ((LocalPosition.at(nextnumber - 1)[0] <10 && LocalPosition.at(nextnumber - 1)[0] > 8) && abs(LocalPosition.at(nextnumber - 1)[1]) < 5)
							{
								std::cout << "！！！！！！！！！！！！！！" << std::endl;
								float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.09, 0, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
								client.moveByAngleThrottle(0, 0.02*coff, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}

							else if ((LocalPosition.at(nextnumber - 1)[0] >8 && LocalPosition.at(nextnumber - 1)[0] < 10) && abs(LocalPosition.at(nextnumber - 1)[1]) > 5)
							{
								std::cout << "。。。。。。。。。。。。。。。。。。。。。。。" << std::endl;
								float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.038, 0, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
								client.moveByAngleThrottle(0, 0.06*coff, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}

							else if ((LocalPosition.at(nextnumber - 1)[0] >10 && LocalPosition.at(nextnumber - 1)[0] < 12) && abs(LocalPosition.at(nextnumber - 1)[1]) > 15)
							{
								std::cout << "................................" << std::endl;
								float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.07, 0, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
								client.moveByAngleThrottle(0, 0.09*coff, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}

							else if ((LocalPosition.at(nextnumber - 1)[0] >10 && LocalPosition.at(nextnumber - 1)[0] < 12) && abs(LocalPosition.at(nextnumber - 1)[1]) < 15)
							{
								std::cout << ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,," << std::endl;
								float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.04, 0.04*coff, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}
							
							else
							{
								std::cout << "something went wrong! " << std::endl;
								client.moveByAngleThrottle(-0.05, 0, 0.65, 0.0f, 0.15f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}
						else
						{
							if (!parking)
							{
								curr_bardata = client.getBarometerdata();
								ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
								ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
								std::cout << "pixel...." << std::endl;
								float pitch = pidP_X.control(xy_temp[0]);
								float roll = pidP_Y.control(xy_temp[1]);
								std::cout << "x_pixel: " << xy_temp[0] << "  y_pixel: " << xy_temp[1] << std::endl;
								float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
								CLIP3(-0.3, pitch, 0.3);
								CLIP3(-0.3, roll, 0.3);
								CLIP3(0.4, delta_throttle, 0.8);
								client.moveByAngleThrottle(-pitch, -roll, delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
							delete[] number;
						}
					}
					else   //////搜索算法
					{
						std::cout << "go up..." << std::endl;
						if (nextnumber == 1 && flag_image)
						{
							client.moveByAngleThrottle(-0.02, 0, 0.5999, 0.0f, 0.15f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						}
						else
						{
							curr_bardata = client.getBarometerdata();
							ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
							ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
							
							if (ned_curr(2) < 10 )
							{
								if (LocalPosition.at(nextnumber - 2)[2] == 1)
								{
									client.moveByAngleThrottle(-0.01, 0, 0.66, 0.0f, 0.25f);
									std::this_thread::sleep_for(std::chrono::duration<double>(0.25f));
								}
								else
								{
									client.moveByAngleThrottle(0, 0, 0.66, 0.0f, 0.25f);
									std::this_thread::sleep_for(std::chrono::duration<double>(0.25f));
								}
								
							}
							else
							{
								client.moveByAngleThrottle(-0.06, 0, 0.63, 0.0f, 0.16f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.16f));
							}
						}
					}
				}
				if (abs(xy_temp[0] - 240) < 18 && abs(xy_temp[1] - 320) <18) count1++;
				if (count1 > 5)
				{
					nextnumber = currentnumber + 1;
					controlmode = 2;
					count1 = 0;
					count_parking = 0;
					count_parking_1 = 0;
					count_parking_2 = 0;
					count_left = 0;
					count_right = 0;
					xy_temp[0] = 0;
					xy_temp[1] = 0;
					flag_parking = false;
					std::cout << "land....." << std::endl;
				}
			}
			else if (controlmode == 2)       //2下降
			{
				clock_t waitstart = clock();
				client.moveByAngleThrottle(0, 0, 0, 0.0f, 0.1f);
				std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));

				clock_t waitend = clock();
				std::cout << "duaration: " << waitend - waitstart << std::endl;
				if (waitend - waitstart > 50)  count2++;
				std::cout << "count2: " << count2 << std::endl;
				if (count2 > 15)
				{
				
					if (nextnumber == 11)
					{
						 ArucoBegin(2) = 11;
						 pidZ.setPoint(ArucoBegin(2), 0.3, 0, 0.4);
						 controlmode = 6;
					}
					else
					{
						controlmode = 0;
						count2 = 0;
						point_control_bardata = client.getBarometerdata();
						ned_target(2) = 6;
						pidX.setPoint(LocalPosition.at(nextnumber - 1)[0], 0.01, 0, 0);
						pidY.setPoint(-LocalPosition.at(nextnumber - 1)[1], 0.01, 0, 0);
						pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
					}															
					std::cout << "take off....." << std::endl;
				}
			}
			else if (controlmode == 3)  //下一目标为障碍圈 
			{
				Vec4i _UpRight;//x,y,a,b
				cv::Mat3b image_front = img2mat.get_front_mat();
				//读深度图，并二值化
				std::vector<ImageResponse> response = img2mat.get_depth_mat();
				image_depth_C1 = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC1);
				image_depth = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC3);
				std::cout << "response.at(0).height: " << response.at(0).height << " response.at(0).width" << response.at(0).width << std::endl;
				memcpy(image_depth_C1.data, response.at(0).image_data_float.data(), sizeof(float)*response.at(0).height*response.at(0).width);			//img = cv::imdecode(response.at(0).image_data_float, 1);
				image_depth_C1.convertTo(image_depth_C1, CV_32FC1, 1 / 255.0);
				image_depth = convertTo3Channels(image_depth_C1);

				//检测深度图/场景图中的椭圆
				std::vector<ell::Ellipse> ellipse_rgb = OnImage(image_front);
				std::vector<ell::Ellipse> ellipse_depth = OnImage(image_depth);
				//深度图里是否有椭圆
				cout << "ellipse_depth.size():" << ellipse_depth.size() << endl;
				if (ellipse_depth.size() > 0)
				{
					cv::namedWindow("imageScene");
					cv::imshow("imageScene", image_depth);
					cv::waitKey(1);
					float Sum_x = 0, Sum_y = 0;
					float a = 0, b = 0;
					a = ellipse_depth[0]._a;
					b = ellipse_depth[0]._b;
					//多个取均值
					for (int i = 0; i < ellipse_depth.size(); i++)
					{
						if (a > ellipse_depth[i]._a) a = ellipse_depth[i]._a;
						if (b > ellipse_depth[i]._b) b = ellipse_depth[i]._b;
						Sum_x = Sum_x + ellipse_depth[i]._xc;
						Sum_y = Sum_y + ellipse_depth[i]._yc;
					}
					_UpRight[0] = Sum_y / ellipse_depth.size();
					_UpRight[1] = Sum_x / ellipse_depth.size();

					//如果中心吻合，则直接冲过去
					if (abs(_UpRight[0] - 240) < 50 && abs(_UpRight[1] - 320) < 15)
					{
						int height, width;
						height = _UpRight[0] + 0.5*(a + b);
						width = _UpRight[1] + 0.5*(a + b);
						if (height > 480) height = 480;
						if (width > 640) width = 640;
						
						float Max_distance = 0;
						float ThresholdD = 4;
						for (int iter_num = 1; iter_num < _UpRight[0] * 640 + _UpRight[1]; iter_num++)
						{
							if (response.at(0).image_data_float.at(iter_num) == 255) continue;
							if (Max_distance < response.at(0).image_data_float.at(iter_num) &&
								response.at(0).image_data_float.at(iter_num) <= 5)
								Max_distance = response.at(0).image_data_float.at(iter_num);
						}
						std::cout << "a+b: " << a + b << std::endl;
						float circle_pitch = -0.29;
						
						if (Max_distance < ThresholdD ||
							response.at(0).image_data_float.at((height - 1) * 640 + _UpRight[1]) < ThresholdD ||
							response.at(0).image_data_float.at((_UpRight[0] - 1) * 640 + width - 1) < ThresholdD) count_go++;
						if (count_go > 3)
						{
							std::cout << "gogogogogo....." << std::endl;
							client.moveByAngleThrottle(circle_pitch, 0, 0.6244, 0.0f, 1.2f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.2f));
							client.hover();
							std::this_thread::sleep_for(std::chrono::duration<double>(2.0f));
							currentnumber = nextnumber;
							flag = true;
							if (flag)
							{
								nextnumber = currentnumber + 1;
								circle_middle_flag = false;
								circle_far10_flag = false;
								circle_far7_flag = false;
								circle_far12_flag = false;
								circle_far13_flag = false;
								circle_far14_flag = false;
								circle_far16_flag = false;
								circle_far18_flag = false;
								last_circle = false;
								last_circle_weight = 1;
								count_go = 0;
								Min_distance = 100;
								client.hover();
								//if (0.5 < LocalPosition.at(nextnumber - 1)[2]) //停机坪
								//{
								//	controlmode = 1;
								//	std::cout << "go for parking....." << std::endl;
								//}
								//else if (LocalPosition.at(nextnumber - 1)[2] < 0.5)   //障碍圈
								//{
									//std::cout << "nextnumber: " << nextnumber << std::endl;
									controlmode = 3;
								//}
							}
						}
						else
						{
							client.moveByAngleThrottle(-0.08, 0, 0.580, 0.0f, 0.1f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						}
					}
					else
					{
						// 调整使无人机正对圆心位置
						std::cout << "pixel...." << std::endl;
						float delta_throttle = (pidP_Z.control(_UpRight[0]));
						float roll = pidP_Y.control(_UpRight[1]);
						std::cout << "x_pixel: " << _UpRight[0] << "  y_pixel: " << _UpRight[1] << std::endl;
						CLIP3(-0.3, roll, 0.3);
						CLIP3(0.3, delta_throttle, 0.85);
						client.moveByAngleThrottle(0, -roll, delta_throttle, 0.0f, 0.1f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						std::cout << "roll: " << roll << "  delta_throttle: " << delta_throttle << std::endl;
						
					}
				}

				else
				{
					std::cout << "last_circle_weight: " << last_circle_weight << std::endl;
					std::cout << "can not find circles. " << std::endl;
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
					std::cout << "delta_throttle: " << delta_throttle << std::endl;
					CLIP3(0.56, delta_throttle, 0.62);
					std::cout << " LocalPositionx: " << LocalPosition.at(nextnumber - 1)[0] << " LocalPositiony: " << -LocalPosition.at(nextnumber - 1)[1] << std::endl;
										
					if (LocalPosition.at(nextnumber - 1)[0] < 4)
					{
						std::cout << "aaaaaaaaaaaaaaaaaaaaaaaa" << std::endl;
						float y = LocalPosition.at(nextnumber - 1)[1];

						client.moveByAngleThrottle(0,-( y*0.01), delta_throttle, 0.0f, 0.1f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
					}
					else if (LocalPosition.at(nextnumber - 1)[0] > 4 && LocalPosition.at(nextnumber - 1)[0] < 7)
					{
						std::cout << "ffffffffffffffffffffff" << std::endl;
						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle)
						{
							client.moveByAngleThrottle(0.1,0, 0.6, 0.0f, 1.9f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.9f));
							last_circle = true;
							last_circle_weight = 0.0;
							client.hover();
						}
						else
						{
							if (!circle_middle_flag)
							{
								client.moveByAngleThrottle(-0.1*last_circle_weight, 0, 0.6, 0.0f, 1.75f);
								std::this_thread::sleep_for(std::chrono::duration<double>(1.75f));
								circle_middle_flag = true;
								client.hover();
							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(0, -(y*0.015), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}
					}
					else if (LocalPosition.at(nextnumber - 1)[0] < 10 && (LocalPosition.at(nextnumber - 1)[0]) >7)
					{
						std::cout << "33333333333333333333333" << std::endl;
						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle)
						{
							client.moveByAngleThrottle(0.1, 0, 0.61, 0.0f, 1.9f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.9f));
							last_circle = true;
							last_circle_weight = 0.001;
							client.hover();
						}
						else
						{
							if (!circle_far7_flag)
							{
								client.moveByAngleThrottle(-0.040*last_circle_weight, 0, 0.6, 0.0f, 2.54f);
								std::this_thread::sleep_for(std::chrono::duration<double>(2.54f));
								circle_far7_flag = true;
								client.hover();															
							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(0, -(y*0.015), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}
					}
					else if (LocalPosition.at(nextnumber - 1)[0] < 12 && (LocalPosition.at(nextnumber - 1)[0]) >10)
					{
						std::cout << "44444444444444444444" << std::endl;
						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle)
						{
							client.moveByAngleThrottle(0.09, 0, 0.615, 0.0f, 1.5f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.5f));
							last_circle = true;
							last_circle_weight = 0.01;
							client.hover();
							std::this_thread::sleep_for(std::chrono::duration<double>(1.0f));
						}
						else
						{
							if (!circle_far10_flag)
							{
								std::cout << "last_circle_weight: " << last_circle_weight << std::endl;
								client.moveByAngleThrottle(-0.09*last_circle_weight, 0, 0.6, 0.0f, 2.9f);
								std::this_thread::sleep_for(std::chrono::duration<double>(2.9f));
								circle_far10_flag = true;
								client.hover();
							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(0, -(y*0.015), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}
					}
					else if (LocalPosition.at(nextnumber - 1)[0] < 13 && (LocalPosition.at(nextnumber - 1)[0]) >12)
					{
						std::cout << "66666666666666666666666" << std::endl;						
						if (!circle_far12_flag)
						{
							client.moveByAngleThrottle(-0.07*last_circle_weight, 0, 0.6, 0.0f, 2.90f);
							std::this_thread::sleep_for(std::chrono::duration<double>(2.90f));
							circle_far12_flag = true;
							client.hover();
						}
						else
						{
							float y = LocalPosition.at(nextnumber - 1)[1];
							client.moveByAngleThrottle(0, -(y*0.015), delta_throttle, 0.0f, 0.1f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						}
					}
					else if (LocalPosition.at(nextnumber - 1)[0] < 14 && (LocalPosition.at(nextnumber - 1)[0]) >13)
					{
						std::cout << "77777777777777777777777777" << std::endl;
						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle) //上一个还是障碍圈
						{
							std::cout << "上一个还是障碍圈：" << std::endl;
							client.moveByAngleThrottle(0.1,0, 0.615, 0.0f, 1.9f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.9f));
							last_circle = true;
							last_circle_weight = 0.05;
							client.hover();
						}
						else
						{
							if (!circle_far13_flag)
							{
								client.moveByAngleThrottle(-0.08*last_circle_weight, 0, 0.6, 0.0f, 3.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(3.1f));
								circle_far13_flag = true;
								client.hover();
							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(-0.01, -(y*0.015), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}
					}
					else if (LocalPosition.at(nextnumber - 1)[0] < 16 && (LocalPosition.at(nextnumber - 1)[0]) >14)
					{
						std::cout << "qqqqqqqqqqqqqqqqqqqqqqqqqq" << std::endl;

						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle) //上一个还是障碍圈
						{
							std::cout << "上一个还是障碍圈：" << std::endl;
							client.moveByAngleThrottle(0.1, 0, 0.615, 0.0f, 2.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(2.0f));
							last_circle = true;
							last_circle_weight = 0.05;
							client.hover();
						}
						else
						{
							if (!circle_far14_flag)
							{
								client.moveByAngleThrottle(-0.085*last_circle_weight, 0, 0.58, 0.0f, 4.0f);
								std::this_thread::sleep_for(std::chrono::milliseconds(4000));
								circle_far14_flag = true;
								client.hover();
							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(0, -(y*0.010), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}
					}

					else if (LocalPosition.at(nextnumber - 1)[0] < 18 && (LocalPosition.at(nextnumber - 1)[0]) >16)
					{
						std::cout << "8888888888888888888888888888" << std::endl;
						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle) //上一个还是障碍圈
						{
							std::cout << "上一个还是障碍圈：" << std::endl;
							client.moveByAngleThrottle(0.1, 0.01, 0.615, 0.0f, 2.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(2.0f));
							last_circle = true;
							last_circle_weight = 0.05;
						}
						else
						{
							if (!circle_far18_flag)
							{
								client.moveByAngleThrottle(-0.08*last_circle_weight, 0, 0.6, 0.0f, 3.6f);
								std::this_thread::sleep_for(std::chrono::duration<double>(3.6f));
								circle_far18_flag = true;
								client.hover();
							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(-0.01, -(y*0.015), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}
					}
					else
					{
						std::cout << "something went wrong! " << std::endl;
						client.moveByAngleThrottle(-0.05, 0, delta_throttle, 0.0f, 0.15f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
					}
					std::cout << "last_circle_weight: " << last_circle_weight << std::endl;
					Min_distance = response.at(0).image_data_float.at(0);
					float pos_circle;
					for (int iter_num = 1; iter_num < response.at(0).image_data_float.size(); iter_num++)
					{
						if (response.at(0).image_data_float.at(iter_num) == 255) continue;
						if (Min_distance > response.at(0).image_data_float.at(iter_num))
						{
							Min_distance = response.at(0).image_data_float.at(iter_num);
							pos_circle = iter_num;
						}

					}
					if (Min_distance < 1.5)
					{
						std::cout << "back ........................" << std::endl;
						client.moveByAngleThrottle(0.09f, 0.0f, 0.60, 0.0f, 0.5f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.5f));						
					}
					if (pos_circle > 640 * 240 && Min_distance < 2.5)
					{
						std::cout << " upup..........." << std::endl;
						client.moveByAngleThrottle(0.0f, 0.0f, 0.60, 0.0f, 0.2f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.2f));
					}
					else if (Min_distance < 2.5 && pos_circle < 640 * 240)
					{
						std::cout << " downdown..........." << std::endl;
						client.moveByAngleThrottle(0.0f, 0.0f, 0.55, 0.0f, 0.2f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.2f));
					}
				}
			}
			else if (controlmode == 4)// 数据收集、地图建模
			{
				/************************** for collect data  ******************************/
				curr_bardata = client.getBarometerdata();
				ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
				ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
				std::cout << "ned_curr: " << ned_curr(2) << std::endl;
				float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
				std::cout << "delta_throttle: " << delta_throttle << std::endl;
				CLIP3(0.4, delta_throttle, 0.8);
				client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
				int i;
				if (abs(ned_target(2) - ned_curr(2)) < 0.08)
				{
					count4++;
				}
				 if (count4 > 30)
				{
					 client.moveByAngleThrottle(-0.1, -0.14, 0.59, 0.0f, 3.0f);
					 std::this_thread::sleep_for(std::chrono::milliseconds(6000));
					 client.moveByAngleThrottle(-0.1, 0, 0.59, 0.0f, 2.0f);//往回飞18.5秒
					 std::this_thread::sleep_for(std::chrono::milliseconds(6000));
					 /*
					client.moveByAngleThrottle(0, 0.2, 0.6125 / (cos(0.2)), 0, 0.5);//右移0.7s
					std::this_thread::sleep_for(std::chrono::milliseconds(500));
					client.moveByAngleThrottle(-0.1, 0, 0.59, 0.0f, 18.5f);
					std::this_thread::sleep_for(std::chrono::milliseconds(1900));
					int j = 1;
					std::string filename = "D://persp";
					for (int i = 1; i <= 17; ++i) {
						auto image = img2mat.get_below_mat();
						ImageForPosition.push_back(image); // Saved image
						std::this_thread::sleep_for(std::chrono::milliseconds(800));
						cv::imwrite(filename + std::to_string(j) + std::string(".jpg"), image);
						std::cout << "no." << i << "    ";
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(3000));
					client.hover();
					std::this_thread::sleep_for(std::chrono::duration<double>(5.0f));
					*/
					//client.moveByAngleThrottle(0, 0.2, 0.6125 / (cos(0.2)), 0, 1.2);//右移1.2s
					//std::this_thread::sleep_for(std::chrono::milliseconds(1200));

					//i = 17;
				//	client.moveByAngleThrottle(0.1, 0, 0.59, 0.0f, 18.5f);//往回飞18.5秒
					//std::this_thread::sleep_for(std::chrono::milliseconds(2000));
				    //j = 2;

					//std::string filename2 = "D://persp2";
					/*for (int i = 1; i <= 17; ++i) {
						auto image2 = img2mat.get_below_mat();
						ImageForPosition2.push_back(image2); // Saved image
						std::this_thread::sleep_for(std::chrono::milliseconds(800));
						cv::imwrite(filename2 + std::to_string(j) + std::string("-2.jpg"), image2);
						std::cout << "2-no." << i << "    ";
					 }
					 std::this_thread::sleep_for(std::chrono::milliseconds(2900));
					 client.hover();
					 std::this_thread::sleep_for(std::chrono::duration<double>(5.0f));*/
					 i = 22;//22是随便弄的，我开心就好
					// LocalPosition[1][2] =  0.1;

				}
				if (i == 22)
				{
					count4 = 0;
					controlmode = 3;
					home_ned(0) = 0;
					home_ned(1) = 0;
					home_ned(2) = 6;
					pidX.setPoint(home_ned(0), 0.001, 0, 0);
					pidY.setPoint(home_ned(1), 0.001, 0, 0);
					pidZ.setPoint(home_ned(2), 0.3, 0, 0.4);
					test_delta_pitch = 0;
				}
			}
			else if (controlmode == 5)//采集完毕
			{
				curr_bardata = client.getBarometerdata();
				ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
				ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
				std::cout << "ned_curr: " << ned_curr(2) << std::endl;
				float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
				std::cout << "delta_throttle: " << delta_throttle << std::endl;
				CLIP3(0.4, delta_throttle, 0.8);
				client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
				if (abs(ned_target(2) - ned_curr(2)) < 0.08)
				{
					count4++;
				}
				if (count4 > 30) {
					client.moveByAngleThrottle(-0.1, 0.1, 0.59, 0.0f, 2.0f);
					std::this_thread::sleep_for(std::chrono::milliseconds(2000));
				}
				controlmode = 3;

				/* std::cout << " collect data complete! " << std::endl;
				if (!FLAG_CB)
				{
					LocalPosition = calculate(25, ImageForPosition, ImageForPosition2);
					if (LocalPosition[10][1] < -6.0)
						ten_position = 2;
					else if (LocalPosition[10][1] < 3.0)
						ten_position = 1;
					client.moveByAngleThrottle(0.1, 0, 0.59, 0.0f, 17.5f);
					std::this_thread::sleep_for(std::chrono::milliseconds(17500));//返回原点
					client.hover();
					client.moveByAngleThrottle(0.2, 0, 0.6125 / (cos(0.2)), 0, 2.0);//左移1.2s，返回原点
					std::this_thread::sleep_for(std::chrono::milliseconds(2000));
					client.hover();
					std::this_thread::sleep_for(std::chrono::duration<double>(3.0f));
					FLAG_CB = true;
				}*/
				/*
				if (abs(ned_curr(2)- home_ned(2))<0.3)
				{
					std::this_thread::sleep_for(std::chrono::duration<double>(1.0f));
					std::cout << " LocalPositionx: " << LocalPosition.at(nextnumber - 1)[0] << " LocalPositiony: " << LocalPosition.at(nextnumber - 1)[1] << std::endl;
					controlmode = 1;
					count_home = 0;
					ned_target(2) = 7;
					pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
					flag_image = true;
				}
				else
				{
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					
					std::cout <<" ned_curr(2): " << ned_curr(2) << std::endl;
					float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
					std::cout <<" delta_throttle: " << delta_throttle << std::endl;
					CLIP3(0.4, delta_throttle, 0.8);
					client.moveByAngleThrottle(0, 0, delta_throttle, 0.0f, 0.0001f);
					std::this_thread::sleep_for(std::chrono::duration<double>(0.0001));
				}*/
			}
			else if (controlmode == 6)
			{
				if (abs(ArucoBegin(2) - ned_curr(2)) < 0.3)
				{
					count_code++;
				}
				if (count_code > 50)
				{
					std::cout << "yaw adjust....." << std::endl;
					Magdata = client.getMagnetometerdata();
					float Mag_y = Magdata.magnetic_field_body.y();
					float yaw = pid_yaw.control(Mag_y);
					std::cout << " yaw: " << yaw << " Mag_y: " << Mag_y << std::endl;
					client.moveByAngleThrottle(0.0f, 0.0f, 0.58999, yaw, 0.01f);
					std::this_thread::sleep_for(std::chrono::duration<double>(0.01f));
					std::cout << "count_yaw: " << count_yaw << std::endl;
					if (abs(Mag_y - target_Mag_y) < 0.008) count_yaw++;
					if (count_yaw > 10 && abs(Mag_y - target_Mag_y) < 0.008)
					{
						if (ten_position == 1)
						{
							client.moveByAngleThrottle(0, 0.25, 0.5888, 0.0f, 1.5f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.5));
							client.hover();
							std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
						}
						if (ten_position == 2)
						{
							client.moveByAngleThrottle(0, 0.25, 0.5888, 0.0f, 3.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(3.0));
							client.hover();
							std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
						}
						client.moveByAngleThrottle(0.25, 0, 0.5888, 0.0f, 1.0f);
						std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
						client.hover();
						std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
						if (!ReadTxt)
						{
							int iterAruco = 0;
							ifstream in;
							in.open("D:\\aruco.txt", ios::in);
							if (in.fail())
							{
								std::cout << "read aruco.txt fail! " << std::endl;
								in.close();
							}
							else
							{
								while (!in.eof() && iterAruco < 5)
								{
									in >> ArucoID[iterAruco];
									iterAruco++;
								}
								in.close();
							}
							ReadTxt = true;
							controlmode = 7;
							count_code = 0;
							count_yaw = 0;
							ned_target(0) = 100;
							ned_target(1) = 320;
							pidP_X.setPoint(ned_target(0), 0.0008, 0, 0.0005);// 这里的x指的是以无人机运动方向为x的反方向
							pidP_Y.setPoint(ned_target(1), 0.0008, 0, 0.0005);
						}
					}
					else
					{
						curr_bardata = client.getBarometerdata();
						ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
						ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
						std::cout << " ned_curr(0): " << ned_curr(0) << " ned_curr(1): " << ned_curr(1) << std::endl;
						float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;

						CLIP3(0.4, delta_throttle, 0.8);
						client.moveByAngleThrottle(0, 0, delta_throttle, 0.0f, 0.01f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
					}
				}
				else
				{
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					
					float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;

					CLIP3(0.4, delta_throttle, 0.8);
					client.moveByAngleThrottle(0, 0, delta_throttle, 0.0f, 0.01f);
					std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
				}
			 
			}	
			else if (controlmode == 7)           //finding the target tree
			{
				std::cout << "TREE_NUM: " << TREE_NUM << std::endl;
				Img = img2mat.get_below_mat();

				if (!Img.empty())
				{
					switch (state)
					{
					case left2right:
						rollCoefficient = 1;
						std::cout << "state: " << "left2right" << std::endl;
						break;

					case right2left:
						rollCoefficient = -1;
						std::cout << "state: " << "right2left" << std::endl;
						statechange = true;
						break;
					default:
						iter_num = 0;
						rollCoefficient = 0;
						pitchCoefficient = 0;
						std::cout << "default " << std::endl;
						break;
					}
					std::vector<Vec2i> tree_circles = detect_tree(Img, ned_curr(2));
					std::cout << "tree circles size：" << tree_circles.size() << std::endl;
					if (tree_circles.size() > 0)
					{
						if (tree_circles.size() == 1)
						{
							int MAX = tree_circles.at(0)[1];
							XY_TREE = tree_circles.at(0);
							for (int i = 0; i < tree_circles.size(); i++)
							{
								if (tree_circles.at(i)[1] > MAX)
								{
									MAX = tree_circles.at(i)[1];
									XY_TREE = tree_circles.at(i);
								}
							}
						}
						else
						{
							int Min = tree_circles.at(0)[0];
							XY_TREE = tree_circles.at(0);
							for (int i = 0; i < tree_circles.size(); i++)
							{
								if (tree_circles.at(i)[0] < Min)
								{
									Min = tree_circles.at(i)[0];
									XY_TREE = tree_circles.at(i);
								}
							}
						}

						if (XY_TREE[0] == 0 && XY_TREE[1] == 0)
						{
							std::cout << "no full circle detected......" << std::endl;
							if (TREE_NUM > 2 && TREE_NUM<7) state = left2right;
							else state = right2left;
							client.moveByAngleThrottle(-0.03, 0.03*rollCoefficient, 0.593333, 0.05f, 0.001f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
						}
						else
						{
							curr_bardata = client.getBarometerdata();
							ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
							ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
							std::cout << "XY_TREE[0]: " << XY_TREE[0] << " XY_TREE[1]: " << XY_TREE[1] << std::endl;
							float pitch = pidP_X.control(XY_TREE[0]);
							float roll = pidP_Y.control(XY_TREE[1]);
							float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
							Magdata = client.getMagnetometerdata();
							float Mag_y = Magdata.magnetic_field_body.y();
							float yaw = pid_yaw.control(Mag_y);
							CLIP3(-0.1, pitch, 0.1);
							CLIP3(-0.1, roll, 0.1);
							CLIP3(0.4, delta_throttle, 0.8);
							std::cout << "pitch: " << pitch << "  roll: " << roll << std::endl;
							client.moveByAngleThrottle(-pitch, -roll, delta_throttle, yaw, 0.1f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						}

					}
					else
					{
						std::cout << "no full circle detected......" << std::endl;

						if (TREE_NUM > 2 && TREE_NUM<8) state = left2right;
						else state = right2left;
						client.moveByAngleThrottle(-0.03, 0.02*rollCoefficient, 0.59333, 0.f, 0.001f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
					}
					if (abs(XY_TREE[0] - 100) < 20 && abs(XY_TREE[1] - 320) < 20) TREE_COUNT++;
					if (TREE_COUNT > 0)
					{
						controlmode = 8;                  // down and then scan ARuco code
						TREE_COUNT = 0;
						XY_TREE[0] = 0;
						XY_TREE[1] = 0;
						ned_target(2) = 3.5;
						pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
						std::cout << "controlmode turn to 8....." << std::endl;
						client.hover();
						std::this_thread::sleep_for(std::chrono::duration<double>(2.0));
					}
				}
			}
			else if (controlmode == 8)   // down and then scan ARuco code
			{
				if (countResult == 5)
				{
					controlmode = 9;
					std::cout << "Aruco code collect end! go to 0. " << std::endl;
					ned_target(2) = 9;
					pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
				}
				if (FLAG_UPDOWN && !is_FLAG_DOWN)  //FLAG_UPDOWN: true, down ;false, up
				{
					std::cout << "down down .............." << std::endl;
					ned_target(2) = 3.5;
					pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
					is_FLAG_DOWN = true;
				}
				else if (!FLAG_UPDOWN && !is_FLAG_UP)
				{
					std::cout << "up up .............." << std::endl;
					ned_target(2) = 11;
					pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
					is_FLAG_UP = true;
				}
				else
				{
					std::cout << "emmmmmmm .............." << std::endl;
				}
				if (abs(ned_target(2) - ned_curr(2)) < 0.2) UPDOWN_COUNT++;
				std::cout << "UPDOWN_COUNT: " << UPDOWN_COUNT << std::endl;
				if (UPDOWN_COUNT > 10)
				{
					if (FLAG_UPDOWN)   //FLAG_UPDOWN: true, down ;false, up
					{
						client.hover();
						std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
						cv::Mat imgCode = img2mat.get_front_mat();
						cv::Mat imgCode1 = img2mat.get_front_mat();
						imwrite("aruco.png", imgCode);
						std::vector< int > ids;
						std::vector<std::vector< Point2f > > corners, rejected;

						aruco::detectMarkers(imgCode, dictionary, corners, ids, detectorParams, rejected);
						std::vector< int > ids1;
						std::vector<std::vector< Point2f > > corners1, rejected1;

						//detect markers and estimate pose
						aruco::detectMarkers(imgCode1, dictionary, corners1, ids1, detectorParams, rejected1);
						if (ids.size() > 0 || ids1.size()>0)
						{
							if (ids.size() > 0)
							{
								aruco::drawDetectedMarkers(imgCode, corners, ids);
								imshow("out", imgCode);
								cv::waitKey(10);
								for (int ids_iter = 0; ids_iter < ids.size(); ids_iter++)
								{
									for (int aruco_iter = 0; aruco_iter < 10; aruco_iter++)
									{
										if (ids.at(ids_iter) == ArucoID[aruco_iter])
										{
											if (result[aruco_iter] == 1)
											{
												std::vector<ImageResponse> responseAruco = img2mat.get_depth_mat();
												SaveResult(imgCode, responseAruco.at(0), corners.at(ids_iter), ArucoID[aruco_iter]);
												result[aruco_iter] = 0;
												countResult++;
												break;
											}
										}
									}
								}
							}
							if (ids1.size() > 0)
							{
								aruco::drawDetectedMarkers(imgCode1, corners1, ids1);
								imshow("out", imgCode1);
								cv::waitKey(10);
								for (int ids_iter = 0; ids_iter < ids1.size(); ids_iter++)
								{
									for (int aruco_iter = 0; aruco_iter < 10; aruco_iter++)
									{
										if (ids1.at(ids_iter) == ArucoID[aruco_iter])
										{
											if (result[aruco_iter] == 1)
											{
												std::vector<ImageResponse> responseAruco = img2mat.get_depth_mat();
												SaveResult(imgCode1, responseAruco.at(0), corners1.at(ids_iter), ArucoID[aruco_iter]);
												result[aruco_iter] = 0;
												countResult++;
												break;
											}
										}
									}
								}
							}
							ned_target(2) = 11;
							pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
							UPDOWN_COUNT = 0;
							TREE_NUM = TREE_NUM + 1;
							FLAG_UPDOWN = false;
							is_FLAG_UP = false;
						}
						else
						{
							ned_target(2) = 11;
							pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
							UPDOWN_COUNT = 0;
							TREE_NUM = TREE_NUM + 1;
							is_FLAG_UP = false;
							FLAG_UPDOWN = false;
						}
					}
					else                     //up and go for next code
					{
						std::cout << "go for next ARuco code......" << std::endl;
						std::cout << "TREE_NUM: " << TREE_NUM << std::endl;
						if (TREE_NUM == 3 || TREE_NUM == 8)
						{
							std::cout << "back a little......" << std::endl;
							client.moveByAngleThrottle(0.45, 0, 0.593333, 0.05f, 1.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
							client.hover();
							std::this_thread::sleep_for(std::chrono::duration<double>(1.5));
						}
						else if (TREE_NUM<3 || TREE_NUM > 8)
						{
							client.moveByAngleThrottle(0, -0.3, 0.59, 0.0f, 1.5f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.8));
						}
						else if ((TREE_NUM > 3 && TREE_NUM < 6) || TREE_NUM == 7)
						{
							client.moveByAngleThrottle(-0.07, 0.3, 0.59, 0.0f, 2.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.8));
						}
						else if (TREE_NUM == 6)
						{
							client.moveByAngleThrottle(-0.3, 0.3, 0.59, 0.0f, 2.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.8));
						}
						else
						{
							controlmode = 9;
						}

						client.hover();
						std::this_thread::sleep_for(std::chrono::duration<double>(1.8));
						controlmode = 7;
						UPDOWN_COUNT = 0;
						FLAG_UPDOWN = true;
						is_FLAG_DOWN = false;
					}
				}
				else
				{
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
					CLIP3(0.4, delta_throttle, 0.8);
					std::cout << "delta_throttle: " << delta_throttle << std::endl;
					client.moveByAngleThrottle(0, 0, delta_throttle, 0.0f, 0.01f);
					std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
				}
			}
			else if (controlmode == 9)
			{
				if (abs(ned_target(2) - ned_curr(2)) < 0.2) HOME_COUNT++;
				if (HOME_COUNT > 10)
				{
					client.moveByAngleThrottle(-0.1, 0, 0.5899999, 0.0f, 17.5f);
					std::this_thread::sleep_for(std::chrono::duration<double>(17.5));
					client.hover();
					std::this_thread::sleep_for(std::chrono::duration<double>(5.0));
					controlmode = 1;
					HOME = true;
					controlmode = 10;
					HOME_COUNT = 0;
					ned_target(0) = 240;
					ned_target(1) = 320;
					pidP_X.setPoint(ned_target(0), 0.0008, 0, 0.0005);// 这里的x指的是以无人机运动方向为x的反方向
					pidP_Y.setPoint(ned_target(1), 0.0008, 0, 0.0005);

				}
				else
				{
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					std::cout << "ned_curr(2): " << ned_curr(2) << std::endl;
					float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
					CLIP3(0.4, delta_throttle, 0.8);
					std::cout << "delta_throttle: " << delta_throttle << std::endl;
					client.moveByAngleThrottle(0, 0, delta_throttle, 0.0f, 0.01f);
					std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
				}
			}
			else if (controlmode == 10)
			{
				image = img2mat.get_below_mat();
				if (!image.empty())
				{
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					std::vector<cv::Mat> mats;
					std::vector<Vec2i> vec = detect_num(image, mats, ned_curr(2));//检测停机坪
					if (vec.size() > 0)
					{
						int size = vec.size();
						std::cout << "there are " << size << " parking boards." << std::endl;
						int *number = new int[size];
						cv::Mat image00;
						cv::Mat test_temp;
						for (int i = 0; i < size; i++)
						{
							image00 = mats[i].clone();
							imshow("test", image00);
							//AffineTransform(mats[i], image00, 255);
							bp = cv::Algorithm::load<cv::ml::ANN_MLP>("numModel.xml");
							Mat destImg; // 
							cvtColor(image00, destImg, CV_BGR2GRAY); // 转为灰度图像
							resize(destImg, test_temp, Size(32, 30), (0, 0), (0, 0), CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
							threshold(test_temp, test_temp, 80, 255, CV_THRESH_BINARY);
							Mat_<float>sampleMat(1, 32 * 30);
							for (int i = 0; i < 32 * 30; ++i)
							{
								sampleMat.at<float>(0, i) = (float)test_temp.at<uchar>(i / 32, i % 32);
							}

							Mat responseMat;
							bp->predict(sampleMat, responseMat);
							Point maxLoc;
							double maxVal = 0;
							minMaxLoc(responseMat, NULL, &maxVal, NULL, &maxLoc);
							cout << "识别结果：" << maxLoc.x << "	相似度:" << maxVal * 100 << "%" << endl;

							number[i] = maxLoc.x;
							if (number[i] == 0 && (maxVal * 100)>65)
							{
								xy_temp = vec.begin()[i];
								break;
							}
						}
						if (xy_temp[0] != 0 && xy_temp[1] != 0)
						{
							curr_bardata = client.getBarometerdata();
							ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
							ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
							std::cout << "pixel...." << std::endl;
							float pitch = pidP_X.control(xy_temp[0]);
							float roll = pidP_Y.control(xy_temp[1]);
							std::cout << "x_pixel: " << xy_temp[0] << "  y_pixel: " << xy_temp[1] << std::endl;
							float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
							CLIP3(-0.3, pitch, 0.3);
							CLIP3(-0.3, roll, 0.3);
							CLIP3(0.4, delta_throttle, 0.8);
							client.moveByAngleThrottle(-pitch, -roll, delta_throttle, 0.0f, 0.1f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						}
						else
						{
							client.moveByAngleThrottle(0, 0.01, 0.58999f, 0, 0.1f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						}
						if (abs(xy_temp[0] - 240) < 18 && abs(xy_temp[1] - 320) < 18)
						{
							client.moveByAngleThrottle(0, 0, 0, 0.0f, 0.1f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							curr_bardata = client.getBarometerdata();
							ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
							ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
							if (ned_curr(2) < 2.5)
							{
								client.moveByAngleThrottle(0, 0, 0, 0.0f, 3.0f);
								std::this_thread::sleep_for(std::chrono::duration<double>(3.0f));
							}
						}
					}
					else
					{
						client.moveByAngleThrottle(0, 0.01, 0.59222, 0, 0.1f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
					}
				}
			}
			clock_t end = clock();
			std::cout << "cost time: " << end - begin << std::endl;
			long time_temp = 50 - (end - begin);
			i_kalman = i_kalman + 1;
			x_kalman = x_kalman + 1;
			y_kalman = x_kalman + 1;
		}
	}
	catch (rpc::rpc_error&  e)
	{
		std::string msg = e.get_error().as<std::string>();
		std::cout << "Exception raised by the API, something went wrong." << std::endl << msg << std::endl;
	}

	return 0;
}