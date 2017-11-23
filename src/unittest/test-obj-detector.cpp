#include <stdexcept>
#include <vector>
#include <random>
#include <glog/logging.h>



#include "Image/EdgeFeature.hpp"
#include "Image/ReadImg.hpp"
#include "Image/ImgProcession.h"
#include "GLRenderer/include/glm.h"
#include "Traker/Config.h"
#include "Traker//Render.h"
#include "Traker/Traker.h"
#include "Traker/Model.h"
#include "tools/OcvYamlConfig.h"
// ORD::Render g_render;
const float mask =5.f;

void init_MAIN(int argc, char* argv[],TL::OcvYamlConfig &config,std::vector<std::string> &Frames,float *prePose,int &starframeId,ifstream &gtData ){
  
  google::InitGoogleLogging(argv[0]);
  Config::configInstance().loadConfig(config);
  FLAGS_log_dir=config.text("Output.Directory.LOG_DIR");     
  FLAGS_stderrthreshold = std::lround(config.value_f("LOG_Threshold"));  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3


  gtData=ifstream(Config::configInstance().gtFile);    
  
    /*** load first frame to init***/
  
  Frames = ED::FileSys::getFiles(Config::configInstance().videoPath,".png");
  cv::Mat frame = cv::imread(Config::configInstance().videoPath+Frames[0]);
  Config::configInstance().VIDEO_WIDTH = frame.size().width;
  Config::configInstance().VIDEO_HEIGHT = frame.size().height;
  frame.release(); 
  starframeId=Config::configInstance().START_INDEX;


  
}
int main(int argc, char* argv[]) {
     
  if (argc < 2) {
        LOG(ERROR) << "usage  Config_xxx.yaml" << endl;
        return -1;      
  }
  
  TL::OcvYamlConfig config(argv[1]);
  auto imgProcessor = std::make_unique<ED::ImgProcession>();
  std::vector<std::string> Frames;
  float prePose[6];  
  int starframeId ;
  ifstream gtData;  
  
  init_MAIN(argc,argv,config,Frames,prePose,starframeId,gtData);

  auto traker = std::make_unique<OD::Traker>(prePose,true); 
  for(/*auto frameFile :Frames*/int frameId=starframeId;frameId<Frames.size();frameId++){
    int64 time0 = cv::getTickCount();	
    //to do
    {
      auto frameFile=Frames[frameId];
      cv::Mat curFrame = cv::imread(Config::configInstance().videoPath+frameFile);
      std::vector<cv::Mat> curFramePYR;
      imgProcessor->getGussainPYR(curFrame,Config::configInstance().IMG_PYR_NUMBER,curFramePYR);

    }
    int64 time1 = cv::getTickCount();
    printf("fps:%f\n",1.0f/((time1-time0)/cv::getTickFrequency()));

  }
  
  
  return 0;
}
