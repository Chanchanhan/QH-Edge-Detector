#include <fstream>      //C++
#include <iostream>
#include<glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include<Eigen/Dense>
#include<Eigen/Core>
#include<Eigen/Geometry>
#include"sophus/so3.h"
#include"sophus/se3.h"

#include "POSEST/include/posest_edft.h"
#include "POSEST/include/posest.h"
#include "ObjectDetector/EdgeDistanceFieldTraking.h"
#include "edge/EdgeDetector.hpp"

using namespace OD;


EdgeDistanceFieldTraking::EdgeDistanceFieldTraking(const Config& config, const Mat& initPose, bool is_writer)
	:imgHeight( config.height), imgWidth(config.width), 
	nCtrPts(0), nReservedCtrPts(2 * (imgHeight + imgWidth)),m_is_writer(is_writer),m_calibration(config.camCalibration)
{

  m_data.m_model = new Model(config);

}
EdgeDistanceFieldTraking::~EdgeDistanceFieldTraking()
{
  if(m_is_writer)
  {
    m_outPose.close();    
  }
}


void EdgeDistanceFieldTraking::getDist(Mat frame)
{

}

Mat EdgeDistanceFieldTraking::toComputePose(const Mat& prePose, const Mat& preFrame, const Mat& curFrame, const Mat& locations, const int frameId,const GLRenderer &render)
{
  ED::EdgeDetector edgeDetector_;
  cv::Mat distMap;
   edgeDetector_.DealWithFrameAsMRWang(curFrame,distMap);
  double prevPose[6],newPose[6], coarsePose[6]={0};
  for(int i=0;i<6;i++){
    prevPose[i]= (double) prePose.data[i];   
  }
  double K[9] = { m_calibration.fx(), 0, m_calibration.cx(), 0, m_calibration.fy(), m_calibration.cy(), 0, 0, 1 };
//   localChamferMatching(frame,prevPose,render, K,newPose,50);
  int lcmRet = localChamferMatching(distMap, prevPose, render, K, coarsePose, 50); // 0 is ok, -1 is error
//   int lcmRet = localColorMatching(curFrame, preFrame, prevPose, render, K, coarsePose, 50); // 0 is ok, -1 is error

  LOG(INFO)<<"coarse Pose: "<<coarsePose[0]<<" "<<coarsePose[1]<<" "<<coarsePose[2]<<" "<<coarsePose[3]<<" "<<coarsePose[4]<<" "<<coarsePose[5];

  if (lcmRet == 0)
  {
    render.camera.setExtrinsic(coarsePose[0], coarsePose[1], coarsePose[2],coarsePose[3], coarsePose[4], coarsePose[5]);
    memcpy(newPose, coarsePose, 6 * sizeof(double));
    memcpy(prevPose, coarsePose, 6 * sizeof(double));    
  }
  
  
}

int EdgeDistanceFieldTraking::localChamferMatching(const Mat &distMap,const double *prevPose,const GLRenderer &renderer,const double K[9], double *pose,
	const int localSize)
{
	// render depth for fg mask
	Mat fgMask;
	renderer.camera.setExtrinsic(prevPose[0], prevPose[1], prevPose[2],
		prevPose[3], prevPose[4], prevPose[5]);
	renderer.drawMode = 0; //filled
	renderer.bgImgUsed = false;
	renderer.render();

	cv::normalize(renderer.depthMap, fgMask, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::threshold(fgMask, fgMask, 254, 255, CV_THRESH_BINARY_INV); // make sure foreground is white

	vector<vector<Point> > contours;
	cv::findContours(fgMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Point> &ctrPts2D = contours[0];
	int contourPointsNum = ctrPts2D.size();

	// local chamfer matching
	cv::Mat scanScores = cv::Mat::zeros(2 * localSize + 1, 2 * localSize + 1, CV_32FC1);
	for (int i = -localSize; i <= localSize; ++i)
	{
		for (int j = -localSize; j <= localSize; ++j)
		{
			float sum = 0;
			for (int k = 0; k < contourPointsNum; ++k)
			{
				int r = ctrPts2D[k].y + i;
				int c = ctrPts2D[k].x + j;
				if (r >= 0 && r < imgHeight && c >= 0 && c < imgWidth)
				{
					sum += distMap.ptr<float>(r)[c];
				}

			}
			if (contourPointsNum>0)
			{
				scanScores.at<float>(i + localSize, j + localSize) = 1.0*sum / contourPointsNum;
			}
			else
			{
				scanScores.at<float>(i + localSize, j + localSize) = 1e9;
			}
		}
	}

	double minDetScore, maxDetScore;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(scanScores, &minDetScore, &maxDetScore, &minLoc, &maxLoc);

	// estimate pose
	double(*ctrPts3DMem)[3]= (double(*)[3])malloc(contourPointsNum*sizeof(double[3]));
	double(*ctrPts2DMem)[2]= (double(*)[2])malloc(contourPointsNum*sizeof(double[2]));
	
	int validCtrPtsNum = 0, r=0, c=0;
	float tmpX = 0, tmpY = 0, tmpZ = 0;
	for (int k = 0; k < contourPointsNum; ++k)
	{
		r = ctrPts2D[k].y + minLoc.y - localSize;
		c = ctrPts2D[k].x + minLoc.x - localSize;
		if (r >= 0 && r < imgHeight && c >= 0 && c < imgWidth)
		{
			if(renderer.unproject(ctrPts2D[k].x, ctrPts2D[k].y, tmpX, tmpY, tmpZ))
			{
				ctrPts3DMem[validCtrPtsNum][0] = tmpX;
				ctrPts3DMem[validCtrPtsNum][1] = tmpY;
				ctrPts3DMem[validCtrPtsNum][2] = tmpZ;

				ctrPts2DMem[validCtrPtsNum][0] = c;
				ctrPts2DMem[validCtrPtsNum][1] = r;

				validCtrPtsNum++;
			}			
		}
	}

	int noutl, *outidx = NULL;
	int ret = posest(ctrPts2DMem, ctrPts3DMem, validCtrPtsNum, 0.75, (double *)K, pose, 6, POSEST_REPR_ERR_NLN_REFINE, outidx, &noutl, 0);

#if OUT_INTER_RESULTS
	cv::Mat scanScores8u, scanScores8uc3;
	cv::normalize(scanScores, scanScores8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cvtColor(scanScores8u, scanScores8uc3, CV_GRAY2BGR);
	cv::circle(scanScores8uc3, minLoc, 3, cv::Scalar(0, 0, 255));
	cv::imshow("cm score", scanScores8uc3);

	Mat distMap8u, distMap8uc3;
	cv::normalize(distMap, distMap8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cvtColor(distMap8u, distMap8uc3, CV_GRAY2BGR);
	for (int k = 0; k < validCtrPtsNum; ++k)
	{
		int r = ctrPts2DMem[k][1];
		int c = ctrPts2DMem[k][0];
		cv::line(distMap8uc3, cv::Point(c, r), cv::Point(c, r), cv::Scalar(0, 255, 0));
	}

	cv::drawContours(distMap8uc3, contours, 0, cv::Scalar(0, 0, 255));
	cv::imshow("local cm refinement", distMap8uc3);
	cv::waitKey(1);
#endif

	free(ctrPts3DMem);
	free(ctrPts2DMem);
	return ret;
}

int EdgeDistanceFieldTraking::localColorMatching(Mat frame,const Mat prevFrame,const  double *prevPose,const GLRenderer &renderer,const double K[9],
	double *pose, int localSize)
{
	// convert color
	Mat frameGray, prevFrameGray;
	cvtColor(frame, frameGray, CV_BGR2GRAY);
	cvtColor(prevFrame, prevFrameGray, CV_BGR2GRAY);

	// render depth for fg mask
	Mat fgMask;
	renderer.camera.setExtrinsic(prevPose[0], prevPose[1], prevPose[2],
		prevPose[3], prevPose[4], prevPose[5]);
	renderer.drawMode = 0; //filled
	renderer.bgImgUsed = false;
	renderer.render();

	cv::normalize(renderer.depthMap, fgMask, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::threshold(fgMask, fgMask, 254, 255, CV_THRESH_BINARY_INV); // make sure foreground is white

	// reduce the foreground area
	const int dilation_size = 5;
	Mat element = cv::getStructuringElement(cv::MORPH_RECT,
		cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	cv::erode(fgMask, fgMask, element);

	// sample inner points in the foreground
	int imgWidth = fgMask.cols;
	int imgHeight = fgMask.rows;
	int bbxMin = imgWidth, bbyMin = imgHeight, bbxMax =-1, bbyMax = -1;
	for (int i = 0; i < fgMask.rows;++i)
	{
		uchar *rptr = fgMask.ptr<uchar>(i);
		for (int j = 0; j < fgMask.cols; ++j)
		{
			if (rptr[j]!=0)
			{
				if (j < bbxMin)
				{
					bbxMin = j;
				}
				if (j > bbxMax)
				{
					bbxMax = j;
				}
				if (i < bbyMin)
				{
					bbyMin = i;
				}
				if (i > bbyMax)
				{
					bbyMax = i;
				}
			}
		}
	}

	const int innerGridStep = 8;
	vector<Point> innerPts2D;
	vector<uchar> innerPtsGray;
	for (int i = bbyMin; i < bbyMax; i += innerGridStep)
	{
		uchar *dptr = fgMask.ptr<uchar>(i);
		uchar *iptr = prevFrameGray.ptr<uchar>(i);
		for (int j = bbxMin; j < bbxMax; j += innerGridStep)
		{
			if (dptr[j] != 0) // foreground point
			{
					innerPts2D.push_back(Point(j, i));
					innerPtsGray.push_back(iptr[j]);
			}
		}
	}

	// local color matching
	int innerPtsNum = innerPts2D.size();
	cv::Mat scanScores = cv::Mat::zeros(2 * localSize + 1, 2 * localSize + 1, CV_32FC1);
	for (int i = -localSize; i <= localSize; ++i)
	{
		for (int j = -localSize; j <= localSize; ++j)
		{
			float sum = 0;
			for (int k = 0; k < innerPtsNum; ++k)
			{
				int r = innerPts2D[k].y + i;
				int c = innerPts2D[k].x + j;
				if (r >= 0 && r < imgHeight && c >= 0 && c < imgWidth)
				{
					float adiff = abs(frameGray.ptr<uchar>(r)[c]-innerPtsGray[k]);
					sum += adiff;
				}

			}
			if (innerPtsNum > 0)
			{
				scanScores.at<float>(i + localSize, j + localSize) = 1.0*sum / innerPtsNum;
			}
			else
			{
				scanScores.at<float>(i + localSize, j + localSize) = 1e9;
			}
		}
	}

	double minDetScore, maxDetScore;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(scanScores, &minDetScore, &maxDetScore, &minLoc, &maxLoc);

	// estimate pose
	double(*innerPts3DMem)[3] = (double(*)[3])malloc(innerPtsNum*sizeof(double[3]));
	double(*innerPts2DMem)[2] = (double(*)[2])malloc(innerPtsNum*sizeof(double[2]));

	int validPtsNum = 0, r = 0, c = 0;
	float tmpX = 0, tmpY = 0, tmpZ = 0;
	for (int k = 0; k < innerPtsNum; ++k)
	{
		r = innerPts2D[k].y + minLoc.y - localSize;
		c = innerPts2D[k].x + minLoc.x - localSize;
		
		if (r >= 0 && r < imgHeight && c >= 0 && c < imgWidth)
		{
			if (renderer.unproject(innerPts2D[k].x, innerPts2D[k].y, tmpX, tmpY, tmpZ))
			{
				innerPts3DMem[validPtsNum][0] = tmpX;
				innerPts3DMem[validPtsNum][1] = tmpY;
				innerPts3DMem[validPtsNum][2] = tmpZ;

				innerPts2DMem[validPtsNum][0] = c;
				innerPts2DMem[validPtsNum][1] = r;

				validPtsNum++;
			}
		}
	}

	int noutl, *outidx = NULL;
	int ret = posest(innerPts2DMem, innerPts3DMem, validPtsNum, 0.75,(double *) K, pose, 6, POSEST_REPR_ERR_NLN_REFINE, outidx, &noutl, 0);

#if OUT_INTER_RESULTS
	cv::Mat scanScores8u, scanScores8uc3;
	cv::normalize(scanScores, scanScores8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cvtColor(scanScores8u, scanScores8uc3, CV_GRAY2BGR);
	cv::circle(scanScores8uc3, minLoc, 3, cv::Scalar(0, 0, 255));
	cv::imshow("color matching score", scanScores8uc3);

	Mat frameCpy = frame.clone();
	for (int k = 0; k < validPtsNum; ++k)
	{
		int before_r = innerPts2D[k].y;
		int before_c = innerPts2D[k].x;
		cv::circle(frameCpy, cv::Point(before_c, before_r), 1, cv::Scalar(0, 0, 255));

		int after_r = innerPts2DMem[k][1];
		int after_c = innerPts2DMem[k][0];
		cv::circle(frameCpy, cv::Point(after_c, after_r), 1, cv::Scalar(0, 255, 0));

		cv::line(frameCpy, cv::Point(before_c, before_r), cv::Point(after_c, after_r), cv::Scalar(255, 0, 0));
	}
	
	cv::imshow("local color refinement", frameCpy);
	cv::waitKey(1);
#endif

	free(innerPts3DMem);
	free(innerPts2DMem);
	return ret;
}
