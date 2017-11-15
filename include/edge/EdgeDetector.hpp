#ifndef EDGEDETECTOR
#define EDGEDETECTOR

#include <stdlib.h>
#include <stdio.h>
#include <iostream>


#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include<glog/logging.h>
#include"edge/DT.hpp"

using namespace cv;

// using namespace cv::ximgproc;
namespace ED{

class EdgeDetector{

public: 
  EdgeDetector();
  ~EdgeDetector();
  EdgeDetector(std::string modelFilename);
  Mat edgeCanny(Mat src);
  Mat extractEdgeOfImg(Mat src);
  void getDistanceTransform(const Mat &,const float &mask, Mat &dst, Mat& locations);
  Mat toTistanceTransform(Mat src);
  void DealWithFrameAsMRWang(const Mat &src,Mat &dist);
private :
  int edgeThresh ;
  int lowThreshold;
  int  max_lowThreshold ;
  int ratio;
  int kernel_size  ;
  Ptr<ximgproc::StructuredEdgeDetection> pDollar;
};


}
#endif