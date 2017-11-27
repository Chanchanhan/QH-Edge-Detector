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

void ImgProcession::getDistanceTransform(const cv::Mat &src, const float &mask, cv::Mat& dst, cv::Mat& locations)
{
  using namespace std;
  cv::Mat edge =edgeCanny(src);
  cv::cvtColor(edge, edge, CV_BGR2GRAY);  
  edge=~edge;
//   cv::imshow("edge",edge);

  cv::Mat input=cv::Mat::zeros(edge.size(),CV_32FC1);
  edge.convertTo(input,CV_32FC1, 1/*/255.0f*/);	
  vector<float> weights;	
  weights.push_back(mask);	
  weights.push_back(mask);
//   LOG(ERROR)<<"input: "<<std::endl<<input;
  int imageHeight=dst.size().height;
  int imageWidth=dst.size().width;

  distanceTransform(input,dst,locations,weights);
//   cv::imshow("dst",dst);
}
void ImgProcession::getGussainPYR(const cv::Mat& src,const  int& nPYR, std::vector< cv::Mat >& dsts)
{
  cv::Mat tmp=src.clone();
  dsts.push_back(tmp.clone());
  for(int i=1;i<nPYR;++i){
    cv::pyrDown(tmp,tmp,cv::Size(tmp.cols*0.5f, tmp.rows*0.5f));

    dsts.push_back(tmp.clone());
    
  }
}



void ImgProcession::DealWithFrameAsMRWang(const cv::Mat& src, cv::Mat& distMap)
{
  cv::Mat frameGray;
  cvtColor(src, frameGray, CV_BGR2GRAY);
  cv::Mat frameEdge,frameCanny;
  double lowThres = 20, highThres = 60; //20, 60  //50, 100
  cv::blur(frameGray, frameGray, cv::Size(3, 3));
  cv::Canny(frameGray, frameCanny, lowThres, highThres);
  std::cout<<"canny finished\n";
//   distMap=toTistanceTransform(frameCanny);
  cv::imshow("frameCanny",frameCanny);

  cv::distanceTransform(~frameCanny, distMap, CV_DIST_L2, 3);// the distance to zero pixels
  cv::imshow("distMap",distMap);
  cv::waitKey(0);
}

cv::Mat ImgProcession::toTistanceTransform(const  cv::Mat &src)
{
  cv::Mat bw;
     
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
  
  cv::Mat dist;
  cv::distanceTransform(bw, dist, CV_DIST_L2, 5);
  // Normalize the distance image for range = {0.0, 1.0}
  // so we can visualize and threshold it
  normalize(dist, dist, 0, 1., cv::NORM_MINMAX);
  return dist;
  
}

      
 
cv::Mat ImgProcession::edgeCanny(const cv::Mat &src)
{

  cv::Mat  src_gray, dst, detected_edges;
  cv::cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );

  dst.create( src.size(), src.type() );
  cv::blur( src_gray, detected_edges, cv::Size(3,3) );
  cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  dst = cv::Scalar::all(0);
  src.copyTo( dst, detected_edges);
//   cv::cvtColor(dst, dst, CV_BGR2GRAY);
//   cv::cvtColor(dst, dst, CV_GRAY2BGR);

//   cv::imshow( "canny dst", dst );
//   cv::waitKey(0);
  return dst;
}

cv::Mat ImgProcession::extractEdgeOfImg(cv::Mat &src){
    cv::Mat3f fsrc;
    src.convertTo(fsrc, CV_32F, 1.0 / 255.0);
    cv::Mat1f edges;
    pDollar->detectEdges(fsrc, edges);

//      cv::imshow("Edges", edges);
//      cv::waitKey(0);
    return edges;
}
ImgProcession::ImgProcession(std::string modelFilename)
{
  std::cout<<modelFilename<<std::endl;
   pDollar = cv::ximgproc::createStructuredEdgeDetection(modelFilename);
}

ImgProcession::~ImgProcession()
{
  if(!pDollar.empty()){
   pDollar.release();
  }
}