#ifndef DETECTOR_H
#define DETECTOR_H
#include <opencv2/core.hpp>
#include "Traker/Model.h"

namespace DT{
class Detctor{
  explicit Detctor();
  explicit Detctor(const int &_nPYR);
public:
    void  toDetect(const std::vector<cv::Mat>& curFrames,float * bestPose);
private:
    void initBasePose(const int &nRotaions,const int &nDistances,const int &nCornerVertices);
private:
  int nPRY;
  float *basePose[];
  OD::Model* m_model[];  
};
  
}


#endif