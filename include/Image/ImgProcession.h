#ifndef IMG_PROCESSION_H
#define IMG_PROCESSION_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>


#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include<glog/logging.h>
#include"Image/DT.hpp"


// using namespace cv::ximgproc;
namespace ED{

class ImgProcession{

public: 
  ImgProcession();
  ~ImgProcession();
  ImgProcession(std::string modelFilename);
  cv::Mat edgeCanny(const cv::Mat &src);
  cv::Mat extractEdgeOfImg(cv::Mat &src);
  void getDistanceTransform(const cv::Mat &,const float &mask, cv::Mat &dst, cv::Mat& locations);
  cv::Mat toTistanceTransform(const cv::Mat &src);
  void DealWithFrameAsMRWang(const cv::Mat &src,cv::Mat &dist);
  void getGussainPYR(const cv::Mat &src,const int &nPYR,std::vector<cv::Mat> &dsts); 
private :
  int edgeThresh ;
  int lowThreshold;
  int  max_lowThreshold ;
  int ratio;
  int kernel_size  ;
  cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar;
};


}
#endif