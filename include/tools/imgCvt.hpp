#ifndef IMG_CVT_HPP
#define IMG_CVT_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace TL{
_para initial_para() {
    _para o;
    o.name = " ";
    o.alpha = 0.65;
    o.beta = 0.75;
    o.eta = 1;
    o.minScore = 0.01;
    o.maxBoxes = 10000;
    o.edgeMinMag = 0.1;
    o.edgeMergeThr = 0.5;
    o.clusterMinMag = 0.5;
    o.maxAspectRatio = 3;
    o.minBoxArea = 1000;//1600,20000
    o.maxBoxLength = 10000;//250,500
    o.gamma = 2;
    o.kappa = 1.5;
    return o;
}

cv::Mat bwdist(Mat I,uchar radius){
     Mat bw;


     cvtColor(I, bw, CV_BGR2GRAY);
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    imshow("Binary Image", bw);
        
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    cv::MatIterator_<uchar> colorit;
    for(int i=0;i<bw.size().width;i++){
      for(int j=0;j<bw.size().height;j++){
	bw.at<uchar>(i,j)=bw.at<uchar>(i,j)>=radius?1:0; 
      }
    }
//     normalize(dist, dist, 0, 1., cv::NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    waitKey(0);
    return dist;
}

//for float mat
void getadd(Mat I, float *I_data) {
    vector<Mat> Ivec;
    split(I, Ivec);
    for (int i1 = 0; i1 < I.channels(); i1++) {
        for (int j1 = 0; j1 < I.cols; j1++) {
            for (int k1 = 0; k1 < I.rows; k1++) {
                I_data[i1 * I.rows * I.cols + j1 * I.rows + k1] = Ivec[i1].at<float>(k1, j1);
            }
        }
    }
}

//for uint8 mat
void getaddu(Mat I, uint8 *I_data) {
    vector<Mat> Ivec;
    split(I, Ivec);
    for (int i1 = 0; i1 < I.channels(); i1++) {
        for (int j1 = 0; j1 < I.cols; j1++) {
            for (int k1 = 0; k1 < I.rows; k1++) {
                I_data[i1 * I.rows * I.cols + j1 * I.rows + k1] = Ivec[i1].at<uint8>(k1, j1);
            }
        }
    }
}


//for float mat
void fillmat(float *I_data, Mat I) {
    assert(I.channels() == 1 || I.channels() == 2 || I.channels() == 3 || I.channels() == 4);
    if (I.channels() == 4) {
        for (int i1 = 0; i1 < I.channels(); i1++) {
            for (int j1 = 0; j1 < I.cols; j1++) {
                for (int k1 = 0; k1 < I.rows; k1++) {
                    I.at<Vec4f>(k1, j1)[i1] = I_data[i1 * I.rows * I.cols + j1 * I.rows + k1];
                }
            }
        }
    } else if (I.channels() == 3) {
        for (int i1 = 0; i1 < I.channels(); i1++) {
            for (int j1 = 0; j1 < I.cols; j1++) {
                for (int k1 = 0; k1 < I.rows; k1++) {
                    I.at<Vec3f>(k1, j1)[i1] = I_data[i1 * I.rows * I.cols + j1 * I.rows + k1];
                }
            }
        }
    } else if (I.channels() == 2) {
        for (int i1 = 0; i1 < I.channels(); i1++) {
            for (int j1 = 0; j1 < I.cols; j1++) {
                for (int k1 = 0; k1 < I.rows; k1++) {
                    I.at<Vec2f>(k1, j1)[i1] = I_data[i1 * I.rows * I.cols + j1 * I.rows + k1];
                }
            }
        }
    } else {
        for (int j1 = 0; j1 < I.cols; j1++) {
            for (int k1 = 0; k1 < I.rows; k1++) {
                I.at<float>(k1, j1) = I_data[j1 * I.rows + k1];
            }
        }
    }
}


//for uint8 mat
void fillmatu(uint8 *I_data, Mat I) {
    assert(I.channels() == 1 || I.channels() == 2 || I.channels() == 3 || I.channels() == 4);
    if (I.channels() == 4) {
        for (int i1 = 0; i1 < I.channels(); i1++) {
            for (int j1 = 0; j1 < I.cols; j1++) {
                for (int k1 = 0; k1 < I.rows; k1++) {
                    I.at<Vec4b>(k1, j1)[i1] = I_data[i1 * I.rows * I.cols + j1 * I.rows + k1];
                }
            }
        }
    } else if (I.channels() == 3) {
        for (int i1 = 0; i1 < I.channels(); i1++) {
            for (int j1 = 0; j1 < I.cols; j1++) {
                for (int k1 = 0; k1 < I.rows; k1++) {
                    I.at<Vec3b>(k1, j1)[i1] = I_data[i1 * I.rows * I.cols + j1 * I.rows + k1];
                }
            }
        }
    } else if (I.channels() == 2) {
        for (int i1 = 0; i1 < I.channels(); i1++) {
            for (int j1 = 0; j1 < I.cols; j1++) {
                for (int k1 = 0; k1 < I.rows; k1++) {
                    I.at<Vec2b>(k1, j1)[i1] = I_data[i1 * I.rows * I.cols + j1 * I.rows + k1];
                }
            }
        }
    } else {
        for (int j1 = 0; j1 < I.cols; j1++) {
            for (int k1 = 0; k1 < I.rows; k1++) {
                I.at<uint8>(k1, j1) = I_data[j1 * I.rows + k1];
            }
        }
    }
}
}

#endif