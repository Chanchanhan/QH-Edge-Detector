#ifndef _CONFIG_H
#define _CONFIG_H

#include <string>

#include "GLRenderer/include/glm.h"
#include "GLRenderer/include/cvCamera.h"
#include "ObjectDetector/CameraCalibration.h"
namespace OD
{
  class Config
  {
  public:
    Camera camera;
    GLMmodel* model;
    CameraCalibration camCalibration ;
    int width;
    int height;
    double fps;
    bool USE_GT;
    bool START_INDEX;
    float DIST_MASK_SIZE;
    unsigned int MAX_ITERATIN_NUM;
    float THREHOLD_ENERGY;
    
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
    bool  USE_MY_TRANSFORMATION ;
    float MAX_VALIAD_DISTANCE ;
    std::string videoPath;
    std::string objFile;  
    std::string gtFile;  
  };
  static Config G_CONFIG;

}
#endif