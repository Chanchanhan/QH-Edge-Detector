#ifndef EDGEDETECTOR
#define EDGEDETECTOR

#include <stdlib.h>
#include <stdio.h>
#include <iostream>


#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

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

Mat EdgeDetector::edgeCanny(Mat src)
{

  Mat  src_gray, dst, detected_edges;
  cv::cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );

  dst.create( src.size(), src.type() );
  cv::blur( src_gray, detected_edges, Size(3,3) );
  cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
  dst = Scalar::all(0);
  src.copyTo( dst, detected_edges);
  
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