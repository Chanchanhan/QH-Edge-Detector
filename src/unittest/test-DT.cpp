
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "edge/DT.hpp"


using namespace cv;
using namespace std;

int main() {
	Mat t = Mat::ones(Size(4, 6), CV_32FC1) * 10;
	Mat r = Mat::zeros(Size(4, 6), CV_32FC1);
	int _row=t.rows;
	int _col=t.cols;
	
	int* locations=new int[t.rows*t.cols*t.dims];

	// Just to test that weighting is working
	vector<float> weights;
// 	weights.push_back(2);
// 	weights.push_back(2);

	t.at<float>(3, 2) = 1;

	cout << "Input:" << endl;
	cout << t << endl << endl;

	distanceTransform(t, r, locations, weights);

	cout << "Result:" << endl;
	cout << r << endl;

// 	int *locations = (int *) l.data;

	cout << endl;

	/*
	 * Print locations of the minimum values
	 */
	
	
	
	cout << "Locations Y:" << endl;
	for (size_t row = 0; row < _row; ++row) {
		for (size_t col = 0; col < _col; ++col) {
			cout << locations[col + _col * row] << ", ";
		}
		cout << endl;
	}

	cout << endl;

	cout << "Locations X:" << endl;
	for (size_t row = 0; row < _row; ++row) {
		for (size_t col = 0; col < _col; ++col) {
			cout << locations[_row*_col + col + _col * row] << ", ";
		}
		cout << endl;
	}

	return 0;

}
