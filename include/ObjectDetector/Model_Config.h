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
    std::string videoPath;
    std::string filename;    
  };
}

#endif