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
#include "Traker/Particle.h"
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
     
  if (argc < 2) {
        LOG(ERROR) << "usage  Config_xxx.yaml" << endl;
        return -1;      
  }
  
  TL::OcvYamlConfig config(argv[1]);
  auto edgeDetector_ = std::make_unique<ED::ImgProcession>();
  std::vector<std::string> Frames;
  float prePose[6];  
  int starframeId = Config::configInstance().START_INDEX;
  ifstream gtData;  
  
  init_MAIN(argc,argv,config,Frames,prePose,starframeId,gtData);

  

  auto particlesFilter=std::make_unique<OD::ParticleFilter>(Config::configInstance().PARTICLE_NUM, Config::configInstance().PARTICLE_ARParam, Config::configInstance().PARTICLE_USE_AR, Config::configInstance().PARTICLE_NoiseRateLow, Config::configInstance().PARTICLE_NoiseRateHigh);
  
  
  OD::Traker traker(prePose,true); 
  particlesFilter->init(prePose);
  for(/*auto frameFile :Frames*/int frameId=starframeId;frameId<Frames.size();frameId++){
    int64 time0 = cv::getTickCount();	
    //to do
    {
      
      particlesFilter->transit();
     
      auto frameFile=Frames[frameId];
      cv::Mat preFrame;
      cv::Mat curFrame = cv::imread(Config::configInstance().videoPath+frameFile);
      if(frameId==starframeId){
	preFrame=curFrame.clone();
      }
      if(!Config::configInstance().USE_GT){
	  particlesFilter->update(traker, curFrame, preFrame,frameId);

      }
      particlesFilter->resample();

      traker.m_data.m_model->DisplayCV( particlesFilter->m_particles[0].rt,curFrame);
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
	preFrame=curFrame.clone();
    }

    }
    int64 time1 = cv::getTickCount();
    printf("fps:%f\n",1.0f/((time1-time0)/cv::getTickFrequency()));

  }
  
  
  return 0;
}
