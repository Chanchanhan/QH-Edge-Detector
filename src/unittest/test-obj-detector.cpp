#include <stdexcept>
#include <vector>
#include <random>
#include<glog/logging.h>

#include "andres/marray.hxx"
#include "andres/ml/decision-trees.hxx"
#include "edge/EdgeFeature.hpp"
#include "edge/ReadImg.hpp"
#include "edge/EdgeDetector.hpp"
#include "GLRenderer/include/glm.h"
#include "ObjectDetector/Model_Config.h"
#include "ObjectDetector//Render.h"
#include "GLRenderer/include/markerDetector.h"
#include "GLRenderer/include/cvCamera.h"
#include "GLRenderer/include/glRenderer.h"
#include "ObjectDetector/EdgeDistanceFieldTraking.h"


const float mask =10.f;

const bool TO_DETECT_MARKER = false;
int main(int argc, char* argv[]) {
  ORD::Render g_render;
  cv::Mat pose(1,6,CV_32FC1);
  OD::Config config;
//   ED::EdgeDetector edgeDetector_("/home/qqh/projects/RandomForest/model.yml");
  ED::EdgeDetector edgeDetector_;

  	

  /*** init parame*/
  
  	
  float fxy =2666.67;
  float cx = 960.f;
  float cy = 540.f;
  float distortions[5] = { 0.f, 0.f, 0.f, 0.f, 0.f };
  config.camera =Camera(fxy, fxy, cx, cy, distortions);
	
  /*** start Detectorters***/
  
  config.filename = "/home/qqh/DataSets/3D Rigid Tracking from RGB Images Dataset/box/openbox.obj";
  config.videoPath ="/home/qqh/DataSets/3D Rigid Tracking from RGB Images Dataset/box/video1/";
  config.camCalibration = OD::CameraCalibration(2666.67,2666.67,960,540);
  config.model = glmReadOBJ(const_cast<char*>(config.filename.c_str()));
  pose.at<float>(0,0)=2.31772f; pose.at<float>(0,1)=0.0820299f; pose.at<float>(0,2)=0.681282f;
  pose.at<float>(0,3)=-0.0581884f; pose.at<float>(0,4)=0.0586302f; pose.at<float>(0,5)=1.29788f;
   	 	 		
  // marker configuration and initialize a marker detector
  float markerSize = 9.0f;
  cv::Size2f marker9x9 = cv::Size2f(markerSize, markerSize);
  float marker2ObjectTran[3] = { 15.0f, 0.0f, 0.0f };
  cv::Mat marker2ObjectTranMat = cv::Mat::zeros(3, 1, CV_32FC1);
  for (int i = 0; i < 3;++i)
    marker2ObjectTranMat.ptr<float>(i)[0] = marker2ObjectTran[i];
  MarkerDetector markerDetector(config.camera, marker9x9);

  /*** load first frame to init***/
  
  std::vector<std::string> Frames = ED::FileSys::getFiles(config.videoPath,".png");
  if(Frames.size()==0){
    LOG(ERROR)<<"load Video failed";
  }
  cv::Mat frame = cv::imread(config.videoPath+Frames[0]);
  config.width = frame.size().width;
  config.height = frame.size().height;
  frame.release(); 
  
  // initialize a renderer
  float nearPlane = 1.0f, farPlane = 1000.0f;
  GLRenderer renderer;
  renderer.init(argc, argv, config.width,config.height, nearPlane, farPlane, config.camera, config.model);
  renderer.bgImgUsed = true;
  renderer.drawMode = 0; // 0:fill, 1: wireframe, 2:points
  	
  OD::EdgeDistanceFieldTraking edfTracker(config, pose,false);

  /***load  optimization ***/
  int frameId = 0;
  double rt[6] = { 0 }, final_e = 1E9;

  for(auto frameFile :Frames){
    int64 time0 = cv::getTickCount();
    cv::Mat curFrame = cv::imread(config.videoPath+frameFile);	
    //to detector marker
    if(TO_DETECT_MARKER){
 
//     frameId++;
//     Mat edgeOfFrame;
//     edgeOfFrame=edgeDetector_.edgeCanny(curFrame);
    markerDetector.processFrame(curFrame);
    const std::vector<cv::Mat> &markerTrans = markerDetector.getTransformations();
    if (markerTrans.size() > 0)
    {
      cv::Mat rVec = cv::Mat::zeros(3, 1, CV_32FC1);
      cv::Mat tVec = cv::Mat::zeros(3, 1, CV_32FC1);  
				
      cv::Mat markerTran = markerTrans[0].clone();
      cv::Mat rMat = cv::Mat::zeros(3, 3, CV_32FC1);
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          rMat.ptr<float>(i)[j] = markerTran.ptr<float>(i)[j];
        }
        tVec.ptr<float>(i)[0] = markerTran.ptr<float>(i)[3];
      }
      cv::Rodrigues(rMat, rVec);
      cv::Mat tVecAdded = rMat*marker2ObjectTranMat;
      tVec = tVec + tVecAdded;
      for (int i = 0; i < 3; ++i)
      {
        rt[i] = rVec.at<float>(i, 0);
        rt[i + 3] = tVec.at<float>(i, 0);
      }
      renderer.camera.setExtrinsic(rt[0], rt[1], rt[2], rt[3], rt[4], rt[5]);
      renderer.bgImg = curFrame;
      renderer.bgImgUsed = true;
      renderer.drawMode = 1; // 0:fill, 1: wireframe, 2:points
      renderer.render();
      curFrame = renderer.rgbImg;
      cv::imshow("curFrame",curFrame);
    } 
    
      cv::waitKey(1);
    }
    
   else{    
     Mat distanceFrame,locations;
     edgeDetector_.getDistanceTransform(curFrame,mask,distanceFrame,locations);
     edfTracker.toComputePose(pose, distanceFrame,locations,frameId,renderer);
   }
    int64 time1 = cv::getTickCount();
    printf("fps:%f\n",1.0f/((time1-time0)/cv::getTickFrequency()));

  }
  
  
  return 0;
}
