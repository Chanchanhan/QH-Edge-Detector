#pragma once
#ifndef TYPE_NAME_H
#define TYPE_NAME_H
#include <iostream>

namespace ED{
  enum PathType{
  train,test,eval,out
  
};
enum Split{
  gini,entropy,twoing
  
};
enum RGBG{
  _RGB,
  _D,
  _RGBD
};

}
#endif