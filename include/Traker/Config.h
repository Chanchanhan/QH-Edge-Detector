#ifndef _CONFIG_H
#define _CONFIG_H

#include <string>
#include <glog/logging.h>
#include "GLRenderer/include/glm.h"
#include "GLRenderer/include/cvCamera.h"
#include "Traker/CameraCalibration.h"
#include "tools/OcvYamlConfig.h"
namespace OD
{
  class Config
  {
  public:
    Camera camera;
    GLMmodel* model;
    CameraCalibration camCalibration ;
    float IMG_DIST_MASK_SIZE;
    int IMG_PYR_NUMBER;
    int VIDEO_WIDTH;
    int VIDEO_HEIGHT;
    double fps;
    bool USE_VIDEO;
    bool USE_GT;
    int START_INDEX;
    bool CV_LINE_P2NP;
    float distortions[5] ;
    unsigned int OPTIMIZER_MAX_ITERATIN_NUM;
    float OPTIMIZER_NEASTP_THREHOLD;
    float OPTIMIZER_POINT_THREHOLD;
    float OPTIMIZER_THREHOLD_ENERGY;
    float OPTIMIZER_COARSE_POSE_SIZE;
    float CV_CIRCLE_RADIUS;
    float CV_DRAW_FRAME;
    float DX_SIZE ;
    float NX_LENGTH ;
    float ENERGY_SIZE ;
    float J_SIZE ;
    float LM_STEP ;
    float INIT_LAMDA;
    float SIZE_A ;
    float INF;
    float THREHOLD_DX;
    bool  USE_SOPHUS ;
    bool  USE_PNP;
    bool  USE_MY_TRANSFORMATION ;
    float OPTIMIZER_MAX_EDGE_DISTANCE ;
    bool PARTICLE_USE ;
    float PARTICLE_ROTATION_GAUSSIAN_STD;
    float PARTICLE_TRANSLATION_GAUSSIAN_STD;
    float PARTICLE_USE_AR;
    float PARTICLE_NoiseRateLow;
    float PARTICLE_NoiseRateHigh;
    float PARTICLE_NUM;
    float PARTICLE_ARParam;
//     float PARTICLE_STD_ROT;
//     float PARTICLE_STD_TRA;
    std::string videoPath;
    std::string objFile;  
    std::string gtFile;  
    std::string DISTORTIONS; 
  public:
    void loadConfig(const TL::OcvYamlConfig &config){
      //parameters for init
      {
	
	Config::configInstance().USE_VIDEO = std::lround(config.value_f("USE_VIDEO"))== 1;
	Config::configInstance().objFile = config.text("Input.Directory.Obj");;
	Config::configInstance().videoPath =config.text("Input.Directory.Video");
	Config::configInstance().gtFile= config.text("Input.Directory.GroudTruth");
	Config::configInstance().DISTORTIONS= config.text("Input.Directory.DISTORTIONS");
	Config::configInstance().model = glmReadOBJ(const_cast<char*>(Config::configInstance().objFile.c_str()));
	Config::configInstance().START_INDEX=std::lround(config.value_f("Init_Frame_Index"));
	Config::configInstance().USE_GT=std::lround(config.value_f("USE_GT_DATA"))==1;
	Config::configInstance().VIDEO_HEIGHT=std::lround(config.value_f("VIDEO_HEIGHT"));
	Config::configInstance().VIDEO_WIDTH=std::lround(config.value_f("VIDEO_WIDTH"));
      }
      //parameters for Detect
      {
	Config::configInstance().IMG_DIST_MASK_SIZE=config.value_f("IMG_DIST_MASK_SIZE");
	Config::configInstance().IMG_PYR_NUMBER=std::lround(config.value_f("IMG_PYR_NUMBER"));

      }
      //parameters for optimize
      {
	Config::configInstance().OPTIMIZER_COARSE_POSE_SIZE=config.value_f("OPTIMIZER_COARSE_POSE_SIZE");
	Config::configInstance().OPTIMIZER_NEASTP_THREHOLD=config.value_f("OPTIMIZER_NEASTP_THREHOLD");
	Config::configInstance().OPTIMIZER_POINT_THREHOLD=config.value_f("OPTIMIZER_POINT_THREHOLD");
	Config::configInstance().OPTIMIZER_MAX_ITERATIN_NUM=std::lround(config.value_f("OPTIMIZER_MAX_ITERATIN_NUM"));
	Config::configInstance().OPTIMIZER_THREHOLD_ENERGY=config.value_f("OPTIMIZER_THREHOLD_ENERGY");
	Config::configInstance().DX_SIZE=config.value_f("DX_SIZE");
	Config::configInstance().NX_LENGTH=config.value_f("NX_LENGTH");
	Config::configInstance().ENERGY_SIZE=config.value_f("ENERGY_SIZE");
	Config::configInstance().LM_STEP=config.value_f("LM_STEP");
	Config::configInstance().INIT_LAMDA=config.value_f("INIT_LAMDA");
	Config::configInstance().SIZE_A=config.value_f("SIZE_A");   
	Config::configInstance().THREHOLD_DX=config.value_f("THREHOLD_DX");
	Config::configInstance().USE_PNP=std::lround(config.value_f("USE_PNP"))== 1;;
	Config::configInstance().USE_MY_TRANSFORMATION=std::lround(config.value_f("USE_MY_TRANSFORMATION"))== 1;
	Config::configInstance().OPTIMIZER_MAX_EDGE_DISTANCE=config.value_f("OPTIMIZER_MAX_EDGE_DISTANCE");
	Config::configInstance().J_SIZE=config.value_f("J_SIZE");
	Config::configInstance().USE_SOPHUS=std::lround(config.value_f("USE_SOPHUS"))== 1;;
	Config::configInstance().USE_MY_TRANSFORMATION=std::lround(config.value_f("USE_MY_TRANSFORMATION"))== 1;;
      }
      //parameters for View
      {
	Config::configInstance().CV_CIRCLE_RADIUS=config.value_f("CV_CIRCLE_RADIUS");
	Config::configInstance().CV_LINE_P2NP=std::lround(config.value_f("CV_LINE_P2NP"))== 1;;
	Config::configInstance().CV_DRAW_FRAME=std::lround(config.value_f("CV_DRAW_FRAME"))== 1;;
      }
      //parameters for Particle
      {
	Config::configInstance().PARTICLE_NoiseRateHigh=config.value_f("PARTICLE_NoiseRateHigh");
	Config::configInstance().PARTICLE_NoiseRateLow=config.value_f("PARTICLE_NoiseRateLow");        
	Config::configInstance().PARTICLE_ROTATION_GAUSSIAN_STD=config.value_f("PARTICLE_ROTATION_GAUSSIAN_STD");
	Config::configInstance().PARTICLE_TRANSLATION_GAUSSIAN_STD=(config.value_f("PARTICLE_TRANSLATION_GAUSSIAN_STD"));;
	Config::configInstance().PARTICLE_NUM=std::lround(config.value_f("PARTICLE_NUM"));    
	Config::configInstance().PARTICLE_USE_AR=std::lround(config.value_f("PARTICLE_USE_AR"))== 1;;  
	Config::configInstance().PARTICLE_USE=std::lround(config.value_f("PARTICLE_USE"))== 1;
	Config::configInstance().PARTICLE_ARParam=config.value_f("PARTICLE_ARParam");
// 	Config::configInstance().PARTICLE_STD_ROT=config.value_f("PARTICLE_STD_ROT");
// 	Config::configInstance().PARTICLE_STD_TRA=config.value_f("PARTICLE_STD_TRA");
      }
      //Load Files
      {
	loadFiles();
	Config::configInstance().camCalibration = OD::CameraCalibration(config.value_f("Calib_FX"),config.value_f("Calib_FY"),config.value_f("Calib_CX"),config.value_f("Calib_CY"),Config::configInstance().distortions);

      }
    }
    static Config& configInstance(){
      static Config G_CONFIG;
      return G_CONFIG;
    }
    private:
      void loadFiles(){
	std::string str;
	ifstream _DISTORTIONS=ifstream(Config::configInstance().DISTORTIONS);    
	if(_DISTORTIONS.is_open()){
	  getline(_DISTORTIONS,str) ;    
	  istringstream gt_line(str);
	  int i=0;
	  for (float pos; gt_line >> pos; ++i) {   
		distortions[i] = pos;   
	  }
	}else{
	  memset(distortions,0,5*sizeof(float));
	}
    }
  };
    

}
#endif