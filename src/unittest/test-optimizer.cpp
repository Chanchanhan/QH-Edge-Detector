#include <stdexcept>
#include <vector>
#include <random>
#include <glog/logging.h>



#include "edge/EdgeFeature.hpp"
#include "edge/ReadImg.hpp"
#include "edge/EdgeDetector.hpp"
#include "GLRenderer/include/glm.h"
#include "ObjectDetector/Model_Config.h"
#include "ObjectDetector//Render.h"
#include "ObjectDetector/Optimizer.h"
#include "ObjectDetector/Model.h"

// ORD::Render g_render;
OD::CameraCalibration g_calibration;
#define USE_GT_POSE  0
const float mask =5.f;

void init_MAIN(int argc, char* argv[],OD::Config &config,std::vector<std::string> &Frames,float *prePose,int starframeId,ifstream &gtData ){
  
  
  FLAGS_log_dir="/home/qqh/output/ObjTrackLog/";
  google::InitGoogleLogging(argv[0]);    //
  FLAGS_stderrthreshold = 1;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  
  config.filename = "/home/qqh/DataSets/3D Rigid Tracking from RGB Images Dataset/box/openbox.obj";
  config.videoPath ="/home/qqh/DataSets/3D Rigid Tracking from RGB Images Dataset/box/video1/";
  config.gtFile ="/home/qqh/DataSets/3D Rigid Tracking from RGB Images Dataset/box/video1/poseGT.txt";
  config.camCalibration = OD::CameraCalibration(2666.67,2666.67,960,540);
  config.model = glmReadOBJ(const_cast<char*>(config.filename.c_str()));
  
  g_calibration =config.camCalibration;
   gtData=ifstream(config.gtFile);  

  
    /*** load first frame to init***/
  
  Frames = ED::FileSys::getFiles(config.videoPath,".png");
  cv::Mat frame = cv::imread(config.videoPath+Frames[0]);
  config.width = frame.size().width;
  config.height = frame.size().height;
//   g_render.init(config.camCalibration,config.width,config.height,argc,argv);
  frame.release(); 
  
  int k=0;
  {
    while(k<starframeId){
      string str;
      getline(gtData,str) ;
      k++;

    } 
    float gtPose[6]={0};
    string str,filename;
    getline(gtData,str) ;
    
    istringstream gt_line(str);
    gt_line>>filename;
    int i=0;
    for (float pos; gt_line >> pos; ++i) {   
	  gtPose[i] = pos;   
    }
    LOG(WARNING)<<str<<" getPose"<<" " << gtPose[0]<<" " << gtPose[1]<<" " << gtPose[2]<<" " << gtPose[3]<<" " << gtPose[4]<<" " << gtPose[5];
    memcpy(prePose,gtPose,sizeof(float)*6);
  }
}
int main(int argc, char* argv[]) {
  

  
//   cv::Mat pose(1,6,CV_32FC1);
  OD::Config config;
  ED::EdgeDetector edgeDetector_;
  std::vector<std::string> Frames;
  float prePose[6];
  /*** init parame
  /*** start Detectorters***/

  
  int starframeId = 50;
  
  ifstream gtData/*(config.gtFile)*/;  
  init_MAIN(argc,argv,config,Frames,prePose,starframeId,gtData);

  OD::Optimizer optimizer(config,prePose,true);
  /***load  optimization ***/
 
 
  for(/*auto frameFile :Frames*/int frameId=starframeId;frameId<Frames.size();frameId++){
    int64 time0 = cv::getTickCount();	
    //to do
    {
      auto frameFile=Frames[frameId];
      cv::Mat curFrame = cv::imread(config.videoPath+frameFile);
#if USE_GT_POSE==0
      Mat distanceFrame,locations;
      edgeDetector_.getDistanceTransform(curFrame,mask,distanceFrame,locations);
      optimizer.optimizingLM(prePose,curFrame,distanceFrame,locations,frameId,prePose);
#endif
      optimizer.m_data.m_model->DisplayCV(prePose,curFrame);

      //to test model  , get its point set ,and try to compute energy
      
      imshow("curFrame",curFrame);
      waitKey(1);

#if USE_GT_POSE == 1
      float gtPose[6]={0};
      string str,filename;
      getline(gtData,str) ;
      istringstream gt_line(str);
      gt_line>>filename;
      int i=0;
      for (float pos; gt_line >> pos; ++i) {        
	if(std::isnan(pos)){
	  break;
	}
	gtPose[i] = pos;
      }     
      if(!isnan(gtPose[0])){
	memcpy(prePose,gtPose,sizeof(float)*6);
      }
#endif 
    }
    int64 time1 = cv::getTickCount();
    printf("fps:%f\n",1.0f/((time1-time0)/cv::getTickFrequency()));

  }
  
  
  return 0;
}
