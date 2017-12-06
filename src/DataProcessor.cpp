#include "DataProcessor.h"
#include "Traker/Config.h"
#include "Image/ReadImg.hpp"
#include "GLRenderer/include/glRenderer.h"
#include "Traker/Traker.h"
DataProcessor::DataProcessor(const int &argc,char **argv,const TL::OcvYamlConfig& config)
{
  using namespace TL;
  using namespace OD;
  Config::configInstance().loadConfig(config);
  FLAGS_log_dir=config.text("Output.Directory.LOG_DIR");     
  FLAGS_stderrthreshold = std::lround(config.value_f("LOG_Threshold"));  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3

  
	
//   renderer.init(argc, argv, Config::configInstance().VIDEO_WIDTH, Config::configInstance().VIDEO_HEIGHT, Config::configInstance().CV_NEAR_PLANE, Config::configInstance().CV_FAR_PLANE, Config::configInstance().camera, Config::configInstance().model);	
//   renderer.bgImgUsed = true;	
//   renderer.drawMode = 0; // 0:fill, 1: wireframe, 2:points	

  gtData= ifstream(Config::configInstance().gtFile);    
  
    /*** load first frame to init***/
  
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
    LOG(WARNING)<<" getPose"<<" " << gtPose[0]<<" " << gtPose[1]<<" " << gtPose[2]<<" " << gtPose[3]<<" " << gtPose[4]<<" " << gtPose[5];
    memcpy(prePose,gtPose,sizeof(float)*6);
  }
}
DataProcessor::~DataProcessor()
{

}
// void DataProcessor::doTraking_Frame(const  std::make_shared<OD::Traker> traker, const Mat& frame)
// {
// 
// }
void DataProcessor::doTrakingWithVideo()
{
  using namespace OD;
  int frameId=starframeId;  cv::VideoCapture vc;
    vc.open(Config::configInstance().videoPath);
    if (!vc.isOpened())
    {
        LOG(ERROR)<<("Cannot open camera\n");
        return ;
    }
    Config::configInstance().VIDEO_WIDTH = (int)vc.get(CV_CAP_PROP_FRAME_WIDTH);
    Config::configInstance().VIDEO_HEIGHT = (int)vc.get(CV_CAP_PROP_FRAME_HEIGHT);
  auto traker = std::make_shared<OD::Traker>(prePose,true); 
    cv::Mat curFrame;
  while (vc.read(curFrame))
  {
      cv::Mat frameDrawing = curFrame.clone();

      if(!Config::configInstance().USE_GT){
	    Mat distanceFrame,locations;
	    float finalE2;
	    traker->toTrack(prePose,curFrame,frameId++,renderer, prePose,finalE2);

	    
	}
	
      cv::Mat line_img = cv::Mat::zeros(Config::configInstance().VIDEO_HEIGHT, Config::configInstance().VIDEO_WIDTH , CV_8UC1);
      traker->m_data.m_model->DisplayCV(prePose, cv::Scalar(0, 0, 255),curFrame);
      traker->m_data.m_model->DisplayCV(prePose, cv::Scalar(255, 255, 255),line_img);
      std::vector<std::vector<cv::Point> > contours;      
      cv::findContours(line_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

//       cv::Mat mask_img = cv::Mat::zeros(Config::configInstance().VIDEO_HEIGHT,Config::configInstance().VIDEO_WIDTH, CV_8UC1);
//       cv::drawContours(mask_img, contours, -1, CV_RGB(255, 255, 255), CV_FILLED);
//       LOG(WARNING)<<contours.size();
      if(Config::configInstance().CV_DRAW_FRAME){
	imshow("mask_img",curFrame);
	waitKey(0);
      }
      if(Config::configInstance().USE_GT){
	getSextGtData(prePose);
      }
    }
}
void DataProcessor::doTrakingWithPictures( )
{
  using namespace OD;
  int frameId=starframeId;

  auto traker = std::make_shared<OD::Traker>(prePose,true); 
  std::vector<std::string> Frames = ED::FileSys::getFiles(Config::configInstance().videoPath,".png");
  	

  for(int frameId=starframeId;frameId<Frames.size();frameId++){
    int64 time0 = cv::getTickCount();	
    //to do
    {
      auto frameFile=Frames[frameId];
      cv::Mat curFrame = cv::imread(Config::configInstance().videoPath+frameFile);
      cv::Mat frameDrawing = curFrame.clone();
//       traker->m_data.m_model->DisplayCV(prePose, cv::Scalar(0, 255, 255),curFrame);
      float nextPose[6];
      if(!Config::configInstance().USE_GT){
	Mat distanceFrame,locations;
	float finalE2;
	traker->toTrack2(prePose,curFrame,frameId,renderer,nextPose,finalE2);

      }
      LOG(WARNING)<<"prePose :"<<prePose[0]<<" "<<prePose[1]<<" "<<prePose[2]<<" "<<prePose[3]<<" "<<prePose[4]<<" "<<prePose[5]<<" ";
      traker->m_data.m_model->DisplayCV(prePose, cv::Scalar(0, 255, 255),curFrame);
      LOG(WARNING)<<"nextPose :"<<nextPose[0]<<" "<<nextPose[1]<<" "<<nextPose[2]<<" "<<nextPose[3]<<" "<<nextPose[4]<<" "<<nextPose[5]<<" ";

      traker->m_data.m_model->DisplayCV(nextPose, cv::Scalar(0, 0, 255),curFrame);

//       frameDrawing = renderer.rgbImg;
      //to test model  , get its point set ,and try to compute energy
      if(Config::configInstance().CV_DRAW_FRAME){
	imshow("curFrame",curFrame);
	waitKey(1);
      }

      if(Config::configInstance().USE_GT){
	getSextGtData(prePose);
      }else{
	memcpy(prePose,nextPose,sizeof(float)*6);
      }

    }
    int64 time1 = cv::getTickCount();
    printf("fps:%f\n",1.0f/((time1-time0)/cv::getTickFrequency()));

  }
}

int DataProcessor::doTraking()
{
  
  using namespace OD;

  auto traker = std::make_shared<OD::Traker>(prePose,true); 
  float nearPlane = 1.0f, farPlane = 1000.0f;	
  if(Config::configInstance().USE_VIDEO){
     doTrakingWithVideo();
  }
  else{
    doTrakingWithPictures();
  }
  return 0;
}



void DataProcessor::getSextGtData(float *newPose)
{
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
	  memcpy(newPose,gtPose,sizeof(float)*6);
	}
}
