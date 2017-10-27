#ifndef EDGEDETECTOR
#define EDGEDETECTOR

#include <stdlib.h>
#include <stdio.h>
#include <iostream>


#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include"edge/DT.hpp"
#include<glog/logging.h>
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
private :
  int edgeThresh ;
  int lowThreshold;
  int  max_lowThreshold ;
  int ratio;
  int kernel_size  ;
  Ptr<ximgproc::StructuredEdgeDetection> pDollar;
};

EdgeDetector::EdgeDetector()
{
  edgeThresh = 1;
  max_lowThreshold = 100;
  ratio = 3;
  kernel_size = 3;
  lowThreshold = 20;
}

void EdgeDetector::getDistanceTransform(const Mat &src, const float &mask, Mat& dst, Mat& locations)
{
  cv::Mat edge =edgeCanny(src);
  cv::cvtColor(edge, edge, CV_BGR2GRAY);  
  edge=~edge;
  imshow("edge",edge);

  cv::Mat input=Mat::zeros(edge.size(),CV_32FC1);
  edge.convertTo(input,CV_32FC1, 1/*/255.0f*/);	
  vector<float> weights;	
  weights.push_back(mask);	
  weights.push_back(mask);
//   LOG(ERROR)<<"input: "<<std::endl<<input;
  distanceTransform(input,dst,locations,weights);
//     imshow("dst",dst/255);
//waitKey(0);
//   LOG(ERROR)<<"dst: "<<std::endl<<dst;
}



cv::Mat EdgeDetector::toTistanceTransform(cv::Mat src)
{
  Mat bw;
     
  cvtColor(src, bw, CV_BGR2GRAY);

  cv::threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
  // Perform the distance transform algorithm
   // imshow("Binary Image", bw);
//   for(int i=0;i<bw.size().height;i++){
//     for(int j=0;j<bw.size().width;j++){
//       bw.at<uchar>(i,j)=255-bw.at<uchar>(i,j);
//     }
//   }
  
  bw=~bw;
  
  Mat dist;
  cv::distanceTransform(bw, dist, CV_DIST_L2, 5);
  // Normalize the distance image for range = {0.0, 1.0}
  // so we can visualize and threshold it
  normalize(dist, dist, 0, 1., NORM_MINMAX);
  return dist;
  
}

      
 
Mat EdgeDetector::edgeCanny(Mat src)
{

  Mat  src_gray, dst, detected_edges;
  cv::cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );

  dst.create( src.size(), src.type() );
  cv::blur( src_gray, detected_edges, Size(3,3) );
  cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  dst = Scalar::all(0);
  src.copyTo( dst, detected_edges);
//   cv::cvtColor(dst, dst, CV_BGR2GRAY);
//   cv::cvtColor(dst, dst, CV_GRAY2BGR);

//   imshow( "canny dst", dst );
//   waitKey(0);
  return dst;
}

cv::Mat EdgeDetector::extractEdgeOfImg(Mat src){
    Mat3f fsrc;
    src.convertTo(fsrc, CV_32F, 1.0 / 255.0);
    Mat1f edges;
    pDollar->detectEdges(fsrc, edges);

//      imshow("Edges", edges);
//      waitKey(0);
    return edges;
}
EdgeDetector::EdgeDetector(std::string modelFilename)
{
  std::cout<<modelFilename<<std::endl;
   pDollar = ximgproc::createStructuredEdgeDetection(modelFilename);
}

EdgeDetector::~EdgeDetector()
{
  if(!pDollar.empty()){
   pDollar.release();
  }
}

}
#endif