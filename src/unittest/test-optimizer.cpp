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

void init_MAIN(int argc, char* argv[],TL::OcvYamlConfig &config,std::vector<std::string> &Frames,float *prePose,int &starframeId,ifstream &gtData ){
  
  google::InitGoogleLogging(argv[0]);
  
  FLAGS_log_dir=config.text("Input.Directory.LOG_DIR");     
  FLAGS_stderrthreshold = std::lround(config.value_f("LOG_Threshold"));  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  ////parameters for init
  Config::configInstance().objFile = config.text("Input.Directory.Obj");;
  Config::configInstance().videoPath =config.text("Input.Directory.Video");
  Config::configInstance().gtFile= config.text("Input.Directory.GroudTruth");
  Config::configInstance().camCalibration = OD::CameraCalibration(config.value_f("Calib_FX"),config.value_f("Calib_FY"),config.value_f("Calib_CX"),config.value_f("Calib_CY"));
  Config::configInstance().model = glmReadOBJ(const_cast<char*>(Config::configInstance().objFile.c_str()));
  Config::configInstance().START_INDEX=std::lround(config.value_f("Init_Frame_Index"));
  Config::configInstance().DIST_MASK_SIZE=config.value_f("DIST_MASK_SIZE");
  Config::configInstance().USE_GT=std::lround(config.value_f("USE_GT_DATA"));

  gtData=ifstream(Config::configInstance().gtFile);    
  
    /*** load first frame to init***/
  
  Frames = ED::FileSys::getFiles(Config::configInstance().videoPath,".png");
  cv::Mat frame = cv::imread(Config::configInstance().videoPath+Frames[0]);
  Config::configInstance().width = frame.size().width;
  Config::configInstance().height = frame.size().height;
  frame.release(); 
  starframeId=Config::configInstance().START_INDEX;
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

  //parameters for optimize
  {
    Config::configInstance().MAX_ITERATIN_NUM=std::lround(config.value_f("MAX_ITERATIN_NUM"));
    Config::configInstance().THREHOLD_ENERGY=config.value_f("THREHOLD_ENERGY");
    Config::configInstance().DX_SIZE=config.value_f("DX_SIZE");
    Config::configInstance().NX_LENGTH=config.value_f("NX_LENGTH");
    Config::configInstance().ENERGY_SIZE=config.value_f("ENERGY_SIZE");
    Config::configInstance().LM_STEP=config.value_f("LM_STEP");
    Config::configInstance().INIT_LAMDA=config.value_f("INIT_LAMDA");
    Config::configInstance().SIZE_A=config.value_f("SIZE_A");   
    Config::configInstance().THREHOLD_DX=config.value_f("THREHOLD_DX");
    Config::configInstance().USE_PNP=std::lround(config.value_f("USE_PNP"))== 1;;
    Config::configInstance().USE_MY_TRANSFORMATION=std::lround(config.value_f("USE_MY_TRANSFORMATION"))== 1;
    Config::configInstance().MAX_VALIAD_DISTANCE=config.value_f("MAX_VALIAD_DISTANCE");
    Config::configInstance().J_SIZE=config.value_f("J_SIZE");
    Config::configInstance().USE_SOPHUS=std::lround(config.value_f("USE_SOPHUS"))== 1;;
    Config::configInstance().USE_MY_TRANSFORMATION=std::lround(config.value_f("USE_MY_TRANSFORMATION"))== 1;;
  }
  //parameters for View
  {
    Config::configInstance().CV_CIRCLE_RADIUS=config.value_f("CV_CIRCLE_RADIUS");
    Config::configInstance().CV_LINE_P2NP=std::lround(config.value_f("CV_LINE_P2NP"))== 1;;
  }
}
int main(int argc, char* argv[]) {
     
  if (argc < 2) {
        LOG(ERROR) << "usage  Config_xxx.yaml" << endl;
        return -1;      
  }
  
  TL::OcvYamlConfig config(argv[1]);
  auto edgeDetector_ = std::make_unique<ED::EdgeDetector>();
  std::vector<std::string> Frames;
  float prePose[6];  
  int starframeId = Config::configInstance().START_INDEX;
  ifstream gtData;  
  
  init_MAIN(argc,argv,config,Frames,prePose,starframeId,gtData);

  auto optimizer = std::make_unique<OD::Optimizer>(prePose,true); 
  for(/*auto frameFile :Frames*/int frameId=starframeId;frameId<Frames.size();frameId++){
    int64 time0 = cv::getTickCount();	
    //to do
    {
      auto frameFile=Frames[frameId];
      cv::Mat curFrame = cv::imread(Config::configInstance().videoPath+frameFile);
      if(!Config::configInstance().USE_GT){
	Mat distanceFrame,locations;
	edgeDetector_->getDistanceTransform(curFrame,mask,distanceFrame,locations);
	optimizer->optimizingLM(prePose,curFrame,distanceFrame,locations,frameId,prePose);
      }
      optimizer->m_data.m_model->DisplayCV(prePose,curFrame);

      //to test model  , get its point set ,and try to compute energy
      
      imshow("curFrame",curFrame);
      waitKey(1);

      if(Config::configInstance().USE_GT){
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
    }

    }
    int64 time1 = cv::getTickCount();
    printf("fps:%f\n",1.0f/((time1-time0)/cv::getTickFrequency()));

  }
  
  
  return 0;
}
