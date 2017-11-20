#include <stdexcept>
#include <vector>
#include <random>
#include <glog/logging.h>



#include "edge/EdgeFeature.hpp"
#include "edge/ReadImg.hpp"
#include "edge/EdgeDetector.hpp"
#include "GLRenderer/include/glm.h"
#include "ObjectDetector/Config.h"
#include "ObjectDetector//Render.h"
#include "ObjectDetector/Optimizer.h"
#include "ObjectDetector/Model.h"
#include "tools/OcvYamlConfig.h"
// ORD::Render g_render;
const float mask =5.f;

void init_MAIN(int argc, char* argv[],TL::OcvYamlConfig &config,std::vector<std::string> &Frames,float *prePose,int starframeId,ifstream &gtData ){
  
  
  FLAGS_log_dir=config.text("Input.Directory.LOG_DIR");
  google::InitGoogleLogging(argv[0]);    //
  FLAGS_stderrthreshold = 1;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  
  G_CONFIG.objFile = config.text("Input.Directory.Obj");;
  G_CONFIG.videoPath =config.text("Input.Directory.Video");
  G_CONFIG.gtFile= config.text("Input.Directory.GroudTruth");
  G_CONFIG.camCalibration = OD::CameraCalibration(config.value_f("Calib_FX"),config.value_f("Calib_FY"),config.value_f("Calib_CX"),config.value_f("Calib_CY"));
  G_CONFIG.model = glmReadOBJ(const_cast<char*>(G_CONFIG.objFile.c_str()));
  G_CONFIG.START_INDEX=std::lround(config.value_f("Init_Frame_Index"));
  G_CONFIG.DIST_MASK_SIZE=config.value_f("DIST_MASK_SIZE");
  G_CONFIG.USE_GT=std::lround(config.value_f("USE_GT_DATA"));

  gtData=ifstream(G_CONFIG.gtFile);  

  
  
  
    /*** load first frame to init***/
  
  Frames = ED::FileSys::getFiles(G_CONFIG.videoPath,".png");
  cv::Mat frame = cv::imread(G_CONFIG.videoPath+Frames[0]);
  G_CONFIG.width = frame.size().width;
  G_CONFIG.height = frame.size().height;
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

  //parameter for optimize
  {
    G_CONFIG.MAX_ITERATIN_NUM=std::lround(config.value_f("MAX_ITERATIN_NUM"));
    G_CONFIG.THREHOLD_ENERGY=config.value_f("THREHOLD_ENERGY");
    G_CONFIG.DX_SIZE=config.value_f("DX_SIZE");
    G_CONFIG.NX_LENGTH=config.value_f("NX_LENGTH");
    G_CONFIG.ENERGY_SIZE=config.value_f("ENERGY_SIZE");
    G_CONFIG.LM_STEP=config.value_f("LM_STEP");
    G_CONFIG.INIT_LAMDA=config.value_f("INIT_LAMDA");
    G_CONFIG.SIZE_A=config.value_f("SIZE_A");   
    G_CONFIG.THREHOLD_DX=config.value_f("THREHOLD_DX");
    G_CONFIG.USE_SOPHUS=std::lround(config.value_f("USE_SOPHUS"))== 1;;
    G_CONFIG.USE_MY_TRANSFORMATION=std::lround(config.value_f("USE_MY_TRANSFORMATION"))== 1;
    G_CONFIG.MAX_VALIAD_DISTANCE=config.value_f("MAX_VALIAD_DISTANCE");
    
    G_CONFIG.USE_SOPHUS=std::lround(config.value_f("USE_SOPHUS"))== 1;;
    G_CONFIG.USE_MY_TRANSFORMATION=std::lround(config.value_f("USE_MY_TRANSFORMATION"))== 1;;
  }
}
int main(int argc, char* argv[]) {
     
  if (argc < 2) {
        LOG(ERROR) << "usage  Config_xxx.yaml" << endl;
        return -1;      
  }
  
   TL::OcvYamlConfig config(argv[1]);
//   cv::Mat pose(1,6,CV_32FC1);
  ED::EdgeDetector edgeDetector_;
  std::vector<std::string> Frames;
  float prePose[6];
  /*** init parame
  /*** start Detectorters***/

  
  int starframeId = G_CONFIG.START_INDEX;
  
  ifstream gtData/*(config.gtFile)*/;  
  init_MAIN(argc,argv,config,Frames,prePose,starframeId,gtData);

  OD::Optimizer optimizer(G_CONFIG,prePose,true);
  /***load  optimization ***/
 
 
  for(/*auto frameFile :Frames*/int frameId=starframeId;frameId<Frames.size();frameId++){
    int64 time0 = cv::getTickCount();	
    //to do
    {
      auto frameFile=Frames[frameId];
      cv::Mat curFrame = cv::imread(G_CONFIG.videoPath+frameFile);
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
