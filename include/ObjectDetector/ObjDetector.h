#ifndef OBJ_DETECTOR_H
#define OBJ_DETECTOR_H
#include <iostream>

#include <opencv2/highgui.hpp>
#include <edge/PointSet.hpp>
namespace OD{

class ObjDetector{
public:
  float computeEnergy();
  void setPointSet();
  
private:
  ED::PointSet p_Set;
};
  
}





#endif