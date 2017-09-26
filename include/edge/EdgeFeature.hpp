#pragma once
#ifndef EDGE_FEATURE_HPP
#define EDGE_FEATURE_HPP
#include <iostream>

#include <opencv2/highgui.hpp>

#include<opencv2/imgproc.hpp>  
#include<opencv2/core.hpp>  
#include<opencv2/imgproc.hpp>  

#include "edge/TypeName.h"

namespace ED{
class EdgeFeature{
  
public:
  EdgeFeature(){};
  EdgeFeature(cv::Mat _Img,RGBG rgbd);
  EdgeFeature(const cv::Mat _Img,const RGBG _rgbd,const unsigned int _nChns,const unsigned int _shrink,const unsigned int _nOrients,const unsigned int _grdSmooth);
  bool computeEdgesChns();
  void onResample(const cv::Mat &img,float scale,cv::Mat &Ishrink);
  void convTri(const cv::Mat from, cv::Mat &dst,const int radius,const int s=1,const int nomex=0);
 // void gradientHist(const cv::Mat img,cv::Mat &Gradient);
  void gradientHist(const cv::Mat &Img,cv ::Mat &Hog,const  int N_DIVS=3,const int N_BINS=16);
  void gradientMag(const cv::Mat img,cv::Mat &Gradient,const int channel=0,const int normRad=0,const float normConst=0.005,const int full=0);
  void showMat(std::string name,cv::Mat imgMat);
private:
  cv::Mat Img;//- [h x w x 3] color input image
  cv::Mat *chnsReg;//- [h x w x nChannel] regular output channels
  cv::Mat *chnsSim;//- [h x w x nChannel] self-similarity output channels
  
  //feature parameters:
  unsigned int nOrients  ;// - [4] number of orientations per gradient scale
  unsigned int grdSmooth ;// - [0] radius for image gradient smoothing (using convTri)
  unsigned int chnSmooth;//  - [2] radius for reg channel smoothing (using convTri)
  unsigned int simSmooth;//  - [8] radius for sim channel smoothing (using convTri)
  unsigned int normRad ;//   - [4] gradient normalization radius (see gradientMag)
  unsigned int shrink  ;//   - [2] amount to shrink channels
  unsigned int nCells    ;// - [5] number of self similarity cells
  unsigned int nChns;
  RGBG rgbd     ;//  - [0] 0:RGB, 1:depth, 2:RBG+depth (for NYU data only)  
 
};


EdgeFeature::EdgeFeature
(const cv::Mat _Img,const RGBG _rgbd,const unsigned int _nChns,const unsigned int _shrink,const unsigned int _nOrients,const unsigned int _grdSmooth)
:rgbd(_rgbd),nChns(_nChns),shrink(_shrink),nOrients(_nOrients),grdSmooth(_grdSmooth)
{

   Img = _Img.clone();
  
}
void EdgeFeature::onResample
(const cv::Mat& img, const float scale, cv::Mat& outImg)
{
//   CvSize toSize;
//   toSize.height = img.size().height*scale;
//   toSize.width = img.size().width*scale;
//   cv::resize(img,Ishrink,toSize);
  int inWidth = img.cols;
  int inHeight = img.rows;
  //assert(img.type() == CV_8UC3);
  int nChannels = 3;

  int outWidth = round(inWidth * scale);
  int outHeight = round(inHeight * scale);
  outImg=cv::Mat(outHeight, outWidth,img.type()); //col-major for OpenCV 
  cv::Size outSize = outImg.size();

  cv::resize(img,
	     outImg,
	     outSize,
	     0, //scaleX -- default = outSize.width / img.cols
	     0, //scaleY -- default = outSize.height / img.rows
	     cv::INTER_LINEAR /* use bilinear interpolation */);  
}


// % INPUTS
// %  I      - [hxwxk] input k channel single image
// %  r      - integer filter radius (or any value between 0 and 1)
// %           filter standard deviation is: sigma=sqrt(r*(r+2)/6)
// %  s      - [1] integer downsampling amount after convolving
// %  nomex  - [0] if true perform computation in matlab (for testing/timing)
// %
// % OUTPUTS
// %  J      - [hxwxk] smoothed image
void EdgeFeature::convTri(const cv::Mat from, cv::Mat& dst, const int radius,const int s,const int nomex)
{
  if(from.empty() || !radius&&s==1){
    dst=from;
    return;
  } 

  cv::Mat kern = cv::Mat_<char>::ones(radius,radius);
  cv::filter2D(Img, dst, Img.depth(), kern);  
}
// //定向梯度直方图
// void EdgeFeature::gradientHist(const cv::Mat img, cv::Mat& Gradient)
// {
//   img.convertTo(img, CV_32F, 1/255.0);
//   // Calculate gradients gx, gy
//   cv::Mat gx, gy; 
//   cv::Sobel(img, gx, CV_32F, 1, 0, 1);
//   cv::Sobel(img, gy, CV_32F, 0, 1, 1);
//   //Calculate gradient magnitude and direction (in degrees)
//   cv::Mat mag, angle; 
//   cv::cartToPolar(gx, gy, mag, angle, 1); 
// }
// 

//Input: Grayscale image,Number of bins,Number of cells = N_DIVS*N_DIVS
//Output: HOG features
void EdgeFeature::gradientHist(const cv::Mat &Img,cv ::Mat &Hog,const int N_DIVS,const int N_BINS)
{
  int N_PHOG =N_DIVS*N_DIVS*N_BINS;
  float BIN_RANGE= (2*CV_PI)/N_BINS;
  //cv::Mat Hog;
  Hog = cv::Mat::zeros(1, N_PHOG, CV_32FC1);
  cv::Mat Ix, Iy;
   //Find orientation gradients in x and y directions
      
  cv::Sobel(Img, Ix, CV_16S, 1, 0, 3);   
  cv::Sobel(Img, Iy, CV_16S, 0, 1, 3);  
  int cellx = Img.cols/N_DIVS; 
  int celly = Img.rows/N_DIVS; 
  int img_area = Img.rows * Img.cols;
  for(int m=0; m < N_DIVS; m++)
  {
    for(int n=0; n < N_DIVS; n++)
    {
      for(int i=0; i<cellx; i++)
      {
	for(int j=0; j<celly; j++)
	{
	  float px, py, grad, norm_grad, angle, nth_bin;                    
	  //px = Ix.at(m*cellx+i, n*celly+j);                   
	  //py = Iy.at(m*cellx+i, n*celly+j);                   
	  px = static_cast<float>(Ix.at<int16_t>((m*cellx)+i, (n*celly)+j ));                    
	  py = static_cast<float>(Iy.at<int16_t>((m*cellx)+i, (n*celly)+j ));                    
	  grad = static_cast<float>(std::sqrt(1.0*px*px + py*py));                   
	  norm_grad = grad/img_area;                    
	  //Orientation                    
	  angle = std::atan2(py,px);                  
                    
	  //convert to 0 to 360 (0 to 2*pi)                    
	  if( angle < 0)                        
	    angle+= 2*CV_PI;     
	  //find appropriate bin for angle                    
	  nth_bin = angle/BIN_RANGE;                    
	  //add magnitude of the edges in the hog matrix                    
	  Hog.at<float>(0,(m*N_DIVS +n)*N_BINS + static_cast<int>(angle)) += norm_grad;                  
	}         
      }         
    }      
  }  
    //Normalization
    
    for(int i=0; i< N_DIVS*N_DIVS; i++)
    { 
        float max=0;
        int j;
        for(j=0; j<N_BINS; j++)
        {
            if(Hog.at<float>(0, i*N_BINS+j) > max)
                max = Hog.at<float>(0,i*N_BINS+j);
        }
        for(j=0; j<N_BINS; j++)
            Hog.at<float>(0, i*N_BINS+j)/=max;
    }
}


// % INPUTS
// %  I          - [hxwxk] input k channel single image
// %  channel    - [0] if>0 color channel to use for gradient computation
// %  normRad    - [0] normalization radius (no normalization if 0)
// %  normConst  - [.005] normalization constant
// %  full       - [0] if true compute angles in [0,2*pi) else in [0,pi)
// %
// % OUTPUTS
// %  M          - [hxw] gradient magnitude at each location
// %  O          - [hxw] approximate gradient orientation modulo PI
//梯度信息
void EdgeFeature::gradientMag(const cv::Mat img, cv::Mat& Gradient,const int channel,const int normRad,const float normConst,const int full)
{
  
  /// Generate grad_x and grad_y
  cv::Mat grad_x, grad_y,src_gray;
  cv::Mat abs_grad_x, abs_grad_y;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  // apply a GaussianBlur to our image to reduce the noise ( kernel size = 3 )
  cv::GaussianBlur(img, img, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

  cv::cvtColor( img, src_gray, CV_BGR2GRAY );

  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  cv::Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  cv::Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta,cv:: BORDER_DEFAULT );
  cv::convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, Gradient );

}
void EdgeFeature::showMat(std::string name,cv::Mat imgMat)
{
  cv::namedWindow(name);  
  cv::imshow(name,imgMat);  
  cv::waitKey(); 
  std::getchar();
  cv::destroyWindow(name);
}

//  INPUTS
//  I          - [h x w x 3] color input image
//  opts       - structured edge model options
bool EdgeFeature::computeEdgesChns()
{
  int nType = rgbd+1;
  int k=0;
  cv::Mat *chns = new cv::Mat[nType*2];
 // showMat("initImg",Img);

  for(int i=0;i<nType;i++){
    if(Img.channels()==1){
      cv::cvtColor(Img, Img,CV_BGR2GRAY);  
    }else{
      cv::cvtColor(Img, Img,CV_BGR2Luv);  
    }
     
  //  showMat("img_cv",Img);
    convTri( Img, Img,grdSmooth);
    cv::Mat Ishrink;
    onResample(Img,1.0/shrink,Ishrink);
 
    k=k+1;
    chns[k]=Ishrink.clone();
    for(int j=0,s=1;j<2;j++,s=pow(2,j)){
      cv::Mat I1,Hog,Grd;
      if(s==shrink){
	I1=Ishrink;  
      }else{
	onResample(Img,1/s,I1); 
      }

      convTri( I1, I1,grdSmooth);
      gradientMag( I1,Grd, 0, normRad, .01 );
      showMat("gra",Grd);
     // gradientHist( Grd,Hog, O, std::max(1,shrink/s), nOrients, 0 );
      k=k+1;
      onResample(Grd,s/shrink,chns[k]);
      k=k+1;
      onResample(Hog,std::max(1.0f,(float)s/shrink),chns[k]);
  }
//   chns=cat(3,chns{1:k}); assert(size(chns,3)==opts.nChns);
// chnSm=opts.chnSmooth/shrink; if(chnSm>1), chnSm=round(chnSm); end
// simSm=opts.simSmooth/shrink; if(simSm>1), simSm=round(simSm); end
// chnsReg=convTri(chns,chnSm); chnsSim=convTri(chns,simSm);
  
  }


}
}


#endif
