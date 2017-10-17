#ifndef _CONFIG_H
#define _CONFIG_H

#include <string>

#include "tools//glm.h"
#include "ObjectDetector/CameraCalibration.h"
namespace OD
{
  class Config
  {
  public:
    GLMmodel* model;
    CameraCalibration camCalibration ;
    int width;
    int height;
    double fps;
    std::string videoPath;
    std::string filename;    
  };
}

#endif