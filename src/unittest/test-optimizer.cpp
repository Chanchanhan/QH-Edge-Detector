#include <stdexcept>
#include <vector>
#include <random>
#include <glog/logging.h>


#include "DataProcessor.h"
// ORD::Render g_render;
const float mask =5.f;

int main(int argc, char* argv[]) {
      
  google::InitGoogleLogging(argv[0]);  
  if (argc < 2) {
        LOG(ERROR) << "usage  TrackConfig.yaml \n";
        return -1;      
  }
  TL::OcvYamlConfig config(argv[1]);
  auto dataProcessor = std::make_unique<DataProcessor>(argc,argv,config); 
  dataProcessor->doTraking();
  
  return 0;
}
