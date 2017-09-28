#pragma once
#ifndef IMG_FILE_READ_HPP
#define IMG_FILE_READ_HPP


#include<iostream>
#include <stdlib.h>  
#include <stdio.h>  
#include <string.h>  



#ifdef WIN32  
#include <direct.h>  
#include <io.h>  
#else
#include <unistd.h>  
#include <dirent.h>  
#endif  


#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include "edge/TypeName.h"
#include"andres/ml/decision-trees.hxx"
#include "edge/EdgeFeature.hpp"
namespace ED{
  

 //template<class FEATURE>
class FileSys{
public :
  FileSys(); 
  FileSys(const std::string  trainPath_,const std::string outPath_);
  bool readImg(const PathType path);
  std::vector<std::string>  getFiles(const std::string path)  ;
  bool getFeatureFromImg(cv::Mat img);
  bool toTrainTree();

private:
  
  //(1) model parameters:
  unsigned int imWidth;//- [32] width of image patches
  unsigned int gtWidth;//- [16] width of ground truth patches
    
  //(2) tree parameters:
  unsigned int nPos;//- [5e5] number of positive patches per tree
  unsigned int nNeg;//[5e5] number of negative patches per tree
  unsigned int nTrainImgs;//- [inf] maximum number of images to use for training
  unsigned int nTrees ;//    - [8] number of trees in forest to train
  unsigned int fracFtrs;//   - [1/4] fraction of features to use to train each tree
  unsigned int minCount ;//  - [1] minimum number of data points to allow split
  unsigned int minChild ;//  - [8] minimum number of data points allowed at child nodes
  unsigned int maxDepth;  // - [64] maximum depth of tree
  unsigned int discretize; //- ['pca'] options include 'pca' and 'kmeans'
  unsigned int nSamples ;//  - [256] number of samples for clustering structured labels
  unsigned int nClasses ;//  - [2] number of classes (clusters) for binary splits
  Split split;     // - ['gini'] options include 'gini', 'entropy' and 'twoing'% 
    
  //(3) feature parameters:
  unsigned int nOrients  ;// - [4] number of orientations per gradient scale
  unsigned int grdSmooth ;// - [0] radius for image gradient smoothing (using convTri)
  unsigned int chnSmooth;//  - [2] radius for reg channel smoothing (using convTri)
  unsigned int simSmooth;//  - [8] radius for sim channel smoothing (using convTri)
  unsigned int normRad ;//   - [4] gradient normalization radius (see gradientMag)
  unsigned int shrink  ;//   - [2] amount to shrink channels
  unsigned int nCells    ;// - [5] number of self similarity cells
  RGBG rgbd     ;//  - [0] 0:RGB, 1:depth, 2:RBG+depth (for NYU data only)
    
  //(4) detection parameters (can be altered after training):
  unsigned int stride;//     - [2] stride at which to compute edges
  unsigned int multiscale;//  - [0] if true run multiscale edge detector
  unsigned int sharpen ;//    - [2] sharpening amount (can only decrease after training)
  unsigned int nTreesEval;//  - [4] number of trees to evaluate per location
  unsigned int nThreads;//    - [4] number of threads for evaluation of trees
  unsigned int nms    ;//     - [0] if true apply non-maximum suppression to edges
  //(5) other parameters:
  unsigned int seed ;//       - [1] seed for random stream (for reproducibility)
  unsigned int useParfor ;//  - [0] if true train trees in parallel (memory intensive)
  unsigned int nChns;
  
  float nChnFtrs;
  float nSimFtrs ;
  float nTotFtrs ;
  
  std::string modelDir ;//   - ['models/'] target directory for storing models
  std::string modelFnm;//    - ['model'] model filename
  std::string bsdsDir;//     - ['BSR/BSDS500/data/'] location of BSDS dataset
  std::string trainPath;
  std::string testPath;
  std::string evalPath;
  std::string outPath;
   
   
    
};
// implementation of DecisionNode

/// Constructs a decision node.
/// 
// template<class FEATURE, class LABEL>
inline FileSys::FileSys(const std::string  trainPath_,const std::string outPath_):   trainPath( trainPath_),outPath( outPath_)
{
  imWidth=32;
  gtWidth=16;
  nPos = 5e5;
  nNeg= 5e5;
  shrink=2;
  nOrients =4;
  rgbd =_RGB;
  
  imWidth=round(std::max(gtWidth,imWidth)/shrink/2)*shrink*2;
  
     
  //compute constants and store
  int nChnsGrad=(nOrients+1)*2; 
  int nChnsColor=3;
  if(rgbd==_D) nChnsColor=1; 
  else if(rgbd==2) { 
     nChnsGrad=nChnsGrad*2; 
     nChnsColor=nChnsColor+1; 
  }
    
  nChns = nChnsGrad+nChnsColor; 
  nChnFtrs = imWidth*imWidth*nChns/shrink/shrink;
  nSimFtrs = (nCells*nCells)*(nCells*nCells-1)/2*nChns;
  nTotFtrs = nChnFtrs + nSimFtrs; 
  
}

bool FileSys::toTrainTree()
{
 // std::vector<std::string>  filesNames = getFiles(trainPath);
  
  /***location of ground truth***/

  std::string trnImgDir =  trainPath+"/images/train/";
  std::string trnDepDir = trainPath+ "/depth/train/";
  std::string trnGtDir = trainPath+ "/groundTruth/train/";
  
  std::vector<std::string>  trainImgIds=getFiles(trnImgDir); 

  
  /***extract commonly used options***/
  int imRadius=imWidth/2;
  int gtRadius=gtWidth/2;
  
  /***finalize setup***/
  
  /***collect positive and negative patches and compute features***/

//   std::vector<EdgeFeature> edgeFeatures;
  EdgeFeature edgeFeatures = EdgeFeature(rgbd,nChns,shrink,nOrients,grdSmooth,normRad,simSmooth,chnSmooth);
  {    
    for(std::string trainImgId:trainImgIds){     
      cv::Mat img=cv::imread(trnImgDir+trainImgId,CV_LOAD_IMAGE_COLOR); 
      edgeFeatures.computeEdgesChns(img);
    }
    
  }
}

/** 
 * @function: 获取path目录下的所有img 对应的的patch
 * @param: path PathType类型 
 * @result：vector<vector<FEATURE>>类型 
*/  
bool FileSys::readImg(const PathType path)
{
  std::vector<std::string> filesNames;
  std::string  dir;
  switch (path){
    case train:
      toTrainTree();
      break;
    case test:
      dir = testPath;
      break;
    case eval:
      dir = evalPath;
      break;
    default:
      return false;
  }
  filesNames = getFiles(dir);
  
//   //set train parameters
//   {    
//     for(fileName:filesNames){
//       fileName =  dir+fileName;
//       cv::Mat img=cv::imread(fileName);  
//       
//     }
//   }
}

/** 
 * @function: 获取path目录下的所有文件名 
 * @param: path *char类型 
 * @result：vector<string>类型 
*/  
//ps 这里需要补上正则表达式，先不搞

std::vector<std::string> FileSys::getFiles(const std::string path/*, const string pattern*/)  
{  
   std:: vector<std::string> files;//存放文件名  
  
#ifdef WIN32  
    _finddata_t file;  
    long lf;  
    //输入文件夹路径  
    if ((lf=_findfirst(path.c_str(), &file)) == -1) {  
        std::cout<<path<<" not found!!!"<<endl;  
    } else {  
        while(_findnext(lf, &file) == 0) {  
            //输出文件名  
            //cout<<file.name<<endl;  
            if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)  
                continue;  
            files.push_back(file.name);  
        }  
    }  
    _findclose(lf);  

#else
//#ifdef linux  
    DIR *dir;  
    struct dirent *ptr;  
    char base[1000];  
   
    if ((dir=opendir(path.c_str())) == NULL)  
        {         
	  perror("Open dir error..."); 
	  exit(1);  
        }  
    while ((ptr=readdir(dir)) != NULL)  
    {  
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir  
                continue;  
        else if(ptr->d_type == 8)    ///file  
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);  
            files.push_back(ptr->d_name);  
        else if(ptr->d_type == 10)    ///link file  
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);  
            continue;  
        else if(ptr->d_type == 4)    ///dir  
        {  
            files.push_back(ptr->d_name);  
            /* 
                memset(base,'\0',sizeof(base)); 
                strcpy(base,basePath); 
                strcat(base,"/"); 
                strcat(base,ptr->d_nSame); 
                readFileList(base); 
            */  
        }  
    }  
    closedir(dir);  
#endif  
    //排序，按从小到大排序  
    sort(files.begin(), files.end());  
    return files;  
}  

}
#endif