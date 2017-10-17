#include <stdexcept>
#include <vector>
#include <random>

#include "andres/marray.hxx"
#include "andres/ml/decision-trees.hxx"
#include "edge/EdgeFeature.hpp"
#include "edge/ReadImg.hpp"
#include "edge/EdgeDetector.hpp"
#include "tools/glm.h"
#include "ObjectDetector/Model_Config.h"
#include "ObjectDetector//Render.h"
#include "ObjectDetector/Model.h"

ORD::Render g_render;
OD::CameraCalibration g_calibration;

int main(int argc, char* argv[]) {
  cv::Mat pose(1,6,CV_32FC1);
  OD::Config config;
//   ED::EdgeDetector edgeDetector_("/home/qqh/projects/RandomForest/model.yml");
  ED::EdgeDetector edgeDetector_;

  /*** init parame
  /*** start Detectorters***/
  
/*  pose.at<float>(0,3)=2.31772f; pose.at<float>(0,4)=0.0820299f; pose.at<float>(0,5)=0.681282f;
  pose.at<float>(0,0)=-0.0581884f; pose.at<float>(0,1)=0.0586302f; pose.at<float>(0,2)=1.29788f; */	
  pose.at<float>(0,0)=2.31772f; pose.at<float>(0,1)=0.0820299f; pose.at<float>(0,2)=0.681282f;
  pose.at<float>(0,3)=-0.0581884f; pose.at<float>(0,4)=0.0586302f; pose.at<float>(0,5)=1.29788f; 
  config.filename = "/home/qqh/DataSets/3D Rigid Tracking from RGB Images Dataset/box/openbox.obj";
  config.videoPath ="/home/qqh/DataSets/3D Rigid Tracking from RGB Images Dataset/box/video1/";
  config.camCalibration = OD::CameraCalibration(2666.67,2666.67,960,540);
  config.model = glmReadOBJ(const_cast<char*>(config.filename.c_str()));
  
  g_calibration =config.camCalibration;
  
  /*** load first frame to init***/
  
  std::vector<std::string> Frames = ED::FileSys::getFiles(config.videoPath,".png");
  cv::Mat frame = cv::imread(config.videoPath+Frames[0]);
  config.width = frame.size().width;
  config.height = frame.size().height;
  g_render.init(config.camCalibration,config.width,config.height,argc,argv);
  frame.release(); 
  
  OD::Model model(config);
  
  model.InitPose(pose);
  /***load  optimization ***/
  int frameId = 0;

  for(auto frameFile :Frames){
    int64 time0 = cv::getTickCount();	
    //to do
    {
    cv::Mat curFrame = cv::imread(config.videoPath+frameFile);
    frameId++;
    Mat edgeOfFrame;
    edgeOfFrame=edgeDetector_.edgeCanny(curFrame);
    model.DisplayCV(pose,edgeOfFrame);
    //to test model  , get its point set ,and try to compute energy
     imshow("curFrame",edgeOfFrame);
    waitKey(0);
    
    
    }
    int64 time1 = cv::getTickCount();
    printf("fps:%f\n",1.0f/((time1-time0)/cv::getTickFrequency()));

  }
  
  
  return 0;
}
