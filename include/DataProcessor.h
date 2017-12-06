#ifndef DATA_PROCESSOR_H
#define DATA_PROCESSOR_H
#include <iostream>
#include<fstream>
#include <opencv2/highgui.hpp>
#include "Traker/Traker.h"
#include "tools/OcvYamlConfig.h"
#include "GLRenderer/include/glRenderer.h"
class DataProcessor{
public:
    explicit DataProcessor(const int &argc,char **argv,const TL::OcvYamlConfig& config);
    int doTraking();
    void doDetecting();
    ~DataProcessor();
private:
//   std::ifstream gtData();
//   void doTraking_Frame(const  std::make_shared<OD::Traker> traker, const cv::Mat &frame);

  int starframeId;
  float prePose[6];
  GLRenderer renderer;
  std::ifstream  gtData;
private:
  void getSextGtData(float *newPose);
    void doTrakingWithVideo(  );
  void doTrakingWithPictures(  );

};
#endif