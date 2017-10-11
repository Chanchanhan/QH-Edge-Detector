#pragma once
#ifndef EDGE_FEATURE_HPP
#define EDGE_FEATURE_HPP
#include <iostream>

#include <opencv2/highgui.hpp>

#include<opencv2/imgproc.hpp>  
#include<opencv2/core.hpp>  
#include<opencv2/imgproc.hpp>  

#include "edge/TypeName.h"
#include "tools/rgbConvertMex.hpp"
#include "tools/gradient.hpp"
#include "tools/model.h"
#include "tools/ConvTri.hpp"

namespace ED{

  
class EdgeFeature{
typedef unsigned int  UInt;  
public:
  EdgeFeature(){};
  EdgeFeature(cv::Mat _Img,RGBG rgbd);
  EdgeFeature(const RGBG _rgbd,const UInt _nChns,const UInt _shrink,const UInt _nOrients,const UInt _grdSmooth,const UInt _normRad,const UInt _simSmooth,const UInt _chnSmooth);
  bool computeEdgesChns(cv::Mat Img);
  void onResample(const cv::Mat &img,float scale,cv::Mat &Ishrink);
//   void convTri(const cv::Mat from, cv::Mat &dst,const int radius,const int s=1,const int nomex=0);
 // void gradientHist(const cv::Mat img,cv::Mat &Gradient);
//   void gradientHist(const cv::Mat &Img,cv ::Mat &Hog,const  int N_DIVS=3,const int N_BINS=16);
  void converTo(const cv::Mat &Img,uint flag =0);
//   void gradientMag(const cv::Mat img,cv::Mat &Gradient,const int orientation=2,const int channel=0,const int normRad=0,const float normConst=0.005,const int full=0);
  void showMat(std::string name,cv::Mat imgMat);
  void computeLabels(std::vector<cv::Mat> Imgs);
  std::vector<tuple<cv::Mat, cv::Mat>> & getEdgeChnns(){
    return edgeChnns;
  }
private:
 // cv::Mat Img;//- [h x w x 3] color input image
//   cv::Mat chnsReg;//- [h x w x nChannel] regular output channels
//   cv::Mat chnsSim;//- [h x w x nChannel] self-similarity output channels
  std::vector<tuple<cv::Mat, cv::Mat>>  edgeChnns;
  //feature parameters:
  UInt nOrients  ;// - [4] number of orientations per gradient scale
  UInt grdSmooth ;// - [0] radius for image gradient smoothing (using convTri)
  UInt chnSmooth;//  - [2] radius for reg channel smoothing (using convTri)
  UInt simSmooth;//  - [8] radius for sim channel smoothing (using convTri)
  UInt normRad ;//   - [4] gradient normalization radius (see gradientMag)
  UInt shrink  ;//   - [2] amount to shrink channels
  UInt nChns;
  RGBG rgbd     ;//  - [0] 0:RGB, 1:depth, 2:RBG+depth (for NYU data only)  
  
  
  //label parameters:
  
 
};


EdgeFeature::EdgeFeature
(const RGBG _rgbd,const UInt _nChns,const UInt _shrink,const UInt _nOrients,const UInt _grdSmooth,const UInt _normRad,const UInt _simSmooth,const UInt _chnSmooth)
:rgbd(_rgbd),nChns(_nChns),shrink(_shrink),nOrients(_nOrients),grdSmooth(_grdSmooth),normRad(_normRad),simSmooth(_simSmooth),chnSmooth(_chnSmooth)
{

  
}




void EdgeFeature::onResample
(const cv::Mat& img, const float scale, cv::Mat& outImg)
{
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


void EdgeFeature::showMat(std::string name,cv::Mat imgMat)
{
  cv::namedWindow(name);  
  cv::imshow(name,imgMat);  
  cv::waitKey(); 
  std::getchar();
  cv::destroyWindow(name);
}
void EdgeFeature::computeLabels(std::vector<cv::Mat > Imgs)
{
  int nImg = Imgs.size();
  for (int i=0;i<nImg;i++){
    
    cv::Mat M=Imgs[i]; 
    cv::Mat bw=TL::bwdist(M,3);
    showMat("bw",bw);
//     M(bwdist(M)<gtRadius)=1;
//     [y,x]=find(M.*B); k2=min(length(y),ceil(nPos/nImgs/nGt));
//     rp=randperm(length(y),k2); y=y(rp); x=x(rp);
//     xy=[xy; x y ones(k2,1)*j]; k1=k1+k2; %#ok<AGROW>
//     [y,x]=find(~M.*B); k2=min(length(y),ceil(nNeg/nImgs/nGt));
//     rp=randperm(length(y),k2); y=y(rp); x=x(rp);
//     xy=[xy; x y ones(k2,1)*j]; k1=k1+k2; %#ok<AGROW>
   
     
  }
}

// Convert from rgb to luv
void rgb2luv(cv::Mat Img, cv::Mat &dest, /*float *I, float *J, int n,*/ float nrm ) {
  cv::MatIterator_<cv::Vec3f> colorit, colorend,destrit;
  dest = cv::Mat_<cv::Vec3f> (Img.rows,Img.cols);  
  for(colorit = Img.begin<cv::Vec3f>(),destrit=dest.begin<cv::Vec3f>(), colorend = Img.end<cv::Vec3f>(); colorit != colorend; ++colorit,++destrit)
  {
    float r, g, b, x, y, z, l,u,v;
    b =(float) (*colorit)[0]/*/255.f*/ /*= rand() % 255*/;       //Blue
    g =(float) (*colorit)[1]/*/255.f*/ /*= rand() % 255*/;       //Green
    r =(float) (*colorit)[2]/*/255.f*/ /*= rand() % 255*/;       //Red  

    x = 0.412453 * r + 0.357580 * g + 0.180423 * b;  
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b;  
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b;  
    if (y > 0.008856) l = (float)116.0 * pow(y, (float)1.0 / (float)3.0) - (float)16.0;  
    else l = (float)903.3 * y;  
  
    float sum = x + 15 * y + 3 * z;  
    if(sum != 0) u = 4 * x / sum, v = 9 * y / sum;  
    else u = 4.0, v = (float)9.0 / (float)15.0; 
    (*destrit)[0] = l;  
    (*destrit)[1] = 13 * l * (u - (float)0.19784977571475);  
    (*destrit)[2] = 13 * l * (v - (float)0.46834507665248);  
  }
}



//  INPUTS
//  I          - [h x w x 3] color input image
//  opts       - structured edge model options
bool EdgeFeature::computeEdgesChns(cv::Mat Img)
{
  
  using namespace cv;
//     clock_t begin = clock();
  double a, b;
  int rowsiz = Img.rows;
  int colsiz = Img.cols;
  auto siz = Img.size();
    
//   uint32 shrink =(uint32) shrink;
  int shrinkrowsiz = rowsiz / shrink;
  int shrinkcolsiz = colsiz / shrink;
  Mat chnsReg(shrinkrowsiz, shrinkcolsiz, CV_32FC(nChns));
  Mat chnsSim(shrinkrowsiz, shrinkcolsiz, CV_32FC(nChns));
  int nTypes = 1;
  int k = 0;
  Mat chns(shrinkrowsiz, shrinkcolsiz, CV_32FC(nChns));
  Mat luv_I(rowsiz, colsiz, CV_32FC3);;
  Mat I_shrink;
  luv_I = TL::rgbToLuvu(Img);

  luv_I.convertTo(Img, CV_32FC3);

  luv_I.release();
  
  
  double scale = (double) 1 / shrink;
  resize(Img, I_shrink, Size(), scale, scale);
  Mat *mergemat = new Mat[5];
  mergemat[k] = I_shrink;
  k++;
  //deal with dirrent type of I,but now, we default it as RGB

  for(int t=0;t<nTypes;t++){
    for (int i = 0; i < 2; i++) {
      int s = (int) pow(2, i);
      Mat I1;
      if (s == shrink)
	I1 = I_shrink;
      else
	resize(Img, I1, Size(), 1 / s, 1 / s);
      Mat float_I1;
      I1 = TL::ConvTri(I1, grdSmooth);
      tuple<Mat, Mat> magout;
      magout = TL::gradientMag(I1, 0);
      Mat M, O;
      tie(M, O) = magout;
      TL::gradientMagNorm(M, normRad, 0.01f);
      int binsiz = max(1, (int) shrink / s);
      Mat H;
      H = TL::gradientHist(M, O, binsiz, nOrients, 1);
      Mat M_re, H_re;
      double rescale = (double) s / shrink;
      resize(M, M_re, Size(), rescale, rescale);
      resize(H, H_re, Size(), fmax(1, rescale), fmax(1, rescale));
//       showMat("M",H);

      mergemat[k] = M_re;
      k++;
      mergemat[k] = H_re;
      k++;
      M_re.release();
      H_re.release();
      M.release();
      O.release();
    
    }  
  
  }
  int *from_to = new int[nChns * 2];
  for (int i = 0; i < nChns; i += 1) {
    from_to[2 * i] = i;
    from_to[2 * i + 1] = i;
  }
  mixChannels(mergemat, 5, &chns, 1, from_to, nChns);
  assert(chns.channels() == nChns);
  double chnSm = chnSmooth / shrink;
  double simSm = simSmooth / shrink;
  if (chnSm > 1) chnSm = round(chnSm);
  if (simSm > 1) simSm = round(simSm);
  chnsReg = TL::ConvTri(chns, chnSm);
  chnsSim = TL::ConvTri(chns, simSm);
  edgeChnns.push_back( make_tuple(chnsReg, chnsSim));
  chnsReg.release();
  chnsSim.release();
//    return output;

}
}


#endif
