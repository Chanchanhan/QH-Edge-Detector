#include "Image/ImgProcession.h"
using namespace ED;
ImgProcession::ImgProcession()
{
  edgeThresh = 1;
  max_lowThreshold = 100;
  ratio = 3;
  kernel_size = 3;
  lowThreshold = 20;
}

void ImgProcession::getDistanceTransform(const Mat &src, const float &mask, Mat& dst, Mat& locations)
{
  using namespace std;
  cv::Mat edge =edgeCanny(src);
  cv::cvtColor(edge, edge, CV_BGR2GRAY);  
  edge=~edge;
//   imshow("edge",edge);

  cv::Mat input=Mat::zeros(edge.size(),CV_32FC1);
  edge.convertTo(input,CV_32FC1, 1/*/255.0f*/);	
  vector<float> weights;	
  weights.push_back(mask);	
  weights.push_back(mask);
//   LOG(ERROR)<<"input: "<<std::endl<<input;
  int imageHeight=dst.size().height;
  int imageWidth=dst.size().width;

  distanceTransform(input,dst,locations,weights);
//   imshow("dst",dst);
}


void ImgProcession::DealWithFrameAsMRWang(const Mat& src, Mat& distMap)
{
  Mat frameGray;
  cvtColor(src, frameGray, CV_BGR2GRAY);
  Mat frameEdge,frameCanny;
  double lowThres = 20, highThres = 60; //20, 60  //50, 100
  cv::blur(frameGray, frameGray, cv::Size(3, 3));
  cv::Canny(frameGray, frameCanny, lowThres, highThres);
  std::cout<<"canny finished\n";
//   distMap=toTistanceTransform(frameCanny);
  imshow("frameCanny",frameCanny);

  cv::distanceTransform(~frameCanny, distMap, CV_DIST_L2, 3);// the distance to zero pixels
  imshow("distMap",distMap);
  waitKey(0);
}

cv::Mat ImgProcession::toTistanceTransform(cv::Mat src)
{
  Mat bw;
     
  cvtColor(src, bw, CV_BGR2GRAY);
  cv::adaptiveThreshold(bw,   // Input image
		bw,// Result binary image
		255,         // 
		cv::ADAPTIVE_THRESH_GAUSSIAN_C, //
		cv::THRESH_BINARY_INV, //
		7, //
		7  //
		);

  bw=~bw;
  
  Mat dist;
  cv::distanceTransform(bw, dist, CV_DIST_L2, 5);
  // Normalize the distance image for range = {0.0, 1.0}
  // so we can visualize and threshold it
  normalize(dist, dist, 0, 1., NORM_MINMAX);
  return dist;
  
}

      
 
Mat ImgProcession::edgeCanny(Mat src)
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

cv::Mat ImgProcession::extractEdgeOfImg(Mat src){
    Mat3f fsrc;
    src.convertTo(fsrc, CV_32F, 1.0 / 255.0);
    Mat1f edges;
    pDollar->detectEdges(fsrc, edges);

//      imshow("Edges", edges);
//      waitKey(0);
    return edges;
}
ImgProcession::ImgProcession(std::string modelFilename)
{
  std::cout<<modelFilename<<std::endl;
   pDollar = ximgproc::createStructuredEdgeDetection(modelFilename);
}

ImgProcession::~ImgProcession()
{
  if(!pDollar.empty()){
   pDollar.release();
  }
}