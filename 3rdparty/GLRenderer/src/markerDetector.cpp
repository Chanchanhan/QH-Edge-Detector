#include <iostream>
#include <sstream>

#include "../include/markerDetector.h"
#include "../include/marker.h"
#include "../include/cvCamera.h"

//#define SHOW_DEBUG_IMAGES

template <typename T>
std::string ToString(const T& value)
{
	std::ostringstream stream;
	stream << value;
	return stream.str();
}

float perimeter(const std::vector<cv::Point2f> &a)
{
	float sum = 0, dx, dy;

	for (size_t i = 0; i < a.size(); i++)
	{
		size_t i2 = (i + 1) % a.size();

		dx = a[i].x - a[i2].x;
		dy = a[i].y - a[i2].y;

		sum += sqrt(dx*dx + dy*dy);
	}

	return sum;
}

//////////////////////////////////////////////////////////////////////////////////////////

MarkerDetector::MarkerDetector(const Camera &calibration, const cv::Size2f &markerRealSize)
	: m_minContourLengthAllowed(100)
	, markerSize(105, 105)
{
	camMatrix = calibration.getIntrinsic().clone();
	distCoeff = calibration.getDistorsions().clone();

	bool centerOrigin = true;
	if (centerOrigin)
	{
		// coordiantes system likes this
		//       ^ y
		//       1 
		//       1 
		// ------1----->x
		//       1
		//       1 
		// start from top-left, clockwise
		m_markerCorners3d.push_back(cv::Point3f(-markerRealSize.width / 2, markerRealSize.height / 2, 0));
		m_markerCorners3d.push_back(cv::Point3f(markerRealSize.width / 2, markerRealSize.height / 2, 0));
		m_markerCorners3d.push_back(cv::Point3f(markerRealSize.width / 2, -markerRealSize.height / 2, 0));
		m_markerCorners3d.push_back(cv::Point3f(-markerRealSize.width / 2, -markerRealSize.height / 2, 0));
	}
	else
	{
		// coordiantes system likes this
		// ^ y
		// 1 
		// 1 
		// 1
		// 1
		// 1----------->x
		// start from top-left, clockwise
		m_markerCorners3d.push_back(cv::Point3f(0, markerRealSize.height -1, 0));
		m_markerCorners3d.push_back(cv::Point3f(markerRealSize.width -1 , markerRealSize.height -1, 0));
		m_markerCorners3d.push_back(cv::Point3f(markerRealSize.width -1, 0, 0));
		m_markerCorners3d.push_back(cv::Point3f(0, 0, 0));
	}

	// image coordinates system must be this, the origin is on the top-left
	// 1-------------->x
	// 1
	// 1
	// 1
	// 1
	// v y
	// start from top - left, clockwise
	m_markerCorners2d.push_back(cv::Point2f(0, 0));
	m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, 0));
	m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, markerSize.height - 1));
	m_markerCorners2d.push_back(cv::Point2f(0, markerSize.height - 1));
}

void MarkerDetector::processFrame(const cv::Mat& frame)
{
	std::vector<Marker> markers;
	findMarkers(frame, markers);

	m_transformations.clear();
	for (size_t i = 0; i < markers.size(); i++)
	{
		m_transformations.push_back(markers[i].m_transformation);
	}
}

const std::vector<cv::Mat>& MarkerDetector::getTransformations() const
{
	return m_transformations;
}


bool MarkerDetector::findMarkers(const cv::Mat& frame, std::vector<Marker>& detectedMarkers)
{
	cv::Mat bgraMat = frame.clone();

	// Convert the image to grayscale
	prepareImage(bgraMat, m_grayscaleImage);
	// Make it binary
	performThreshold(m_grayscaleImage, m_thresholdImg);
	// Detect contours
	m_minContourLengthAllowed = m_grayscaleImage.cols / 5;

	findContours(m_thresholdImg, m_contours, m_minContourLengthAllowed);
	// Find closed contours that can be approximated with 4 points
	findCandidates(m_contours, detectedMarkers);

	// Find is them are markers
	recognizeMarkers(m_grayscaleImage, detectedMarkers);

	// Calculate their poses
	estimatePosition(detectedMarkers);

	//sort by id
	std::sort(detectedMarkers.begin(), detectedMarkers.end());
	cv::waitKey(1);
	return true;
}

void MarkerDetector::prepareImage(const cv::Mat& bgraMat, cv::Mat& grayscale)
{
	// Convert to grayscale
	cv::cvtColor(bgraMat, grayscale, CV_BGRA2GRAY);
}

void MarkerDetector::performThreshold(const cv::Mat& grayscale, cv::Mat& thresholdImg)
{
	//cv::threshold(grayscale, thresholdImg, 127, 255, cv::THRESH_BINARY_INV);

	cv::adaptiveThreshold(grayscale,   // Input image
		thresholdImg,// Result binary image
		255,         // 
		cv::ADAPTIVE_THRESH_GAUSSIAN_C, //
		cv::THRESH_BINARY_INV, //
		7, //
		7  //
		);


#ifdef SHOW_DEBUG_IMAGES
	cv::imshow("Threshold image", thresholdImg);
#endif
}

void MarkerDetector::findContours(cv::Mat& thresholdImg, ContoursVector& contours, int minContourPointsAllowed)
{
	ContoursVector allContours;
	cv::findContours(thresholdImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	contours.clear();
	for (size_t i = 0; i < allContours.size(); i++)
	{
		int contourSize = allContours[i].size();
		if (contourSize > minContourPointsAllowed)
		{
			contours.push_back(allContours[i]);
		}
	}

#ifdef SHOW_DEBUG_IMAGES
	{
		cv::Mat contoursImage(thresholdImg.size(), CV_8UC1);
		contoursImage = cv::Scalar(0);
		cv::drawContours(contoursImage, contours, -1, cv::Scalar(255), 2, CV_AA);
		cv::imshow("Contours", contoursImage);
	}
#endif
}

void MarkerDetector::findCandidates
(
const ContoursVector& contours,
std::vector<Marker>& detectedMarkers
)
{
	std::vector<cv::Point>  approxCurve;
	std::vector<Marker>     possibleMarkers;

	// For each contour, analyze if it is a parallelepiped likely to be the marker
	for (size_t i = 0; i < contours.size(); i++)
	{
		// Approximate to a polygon
		double eps = contours[i].size() * 0.05;
		cv::approxPolyDP(contours[i], approxCurve, eps, true);

		// We interested only in polygons that contains only four points
		if (approxCurve.size() != 4)
			continue;

		// And they have to be convex
		if (!cv::isContourConvex(approxCurve))
			continue;

		// Ensure that the distance between consecutive points is large enough
		float minDist = std::numeric_limits<float>::max();

		for (int i = 0; i < 4; i++)
		{
			cv::Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredSideLength = side.dot(side);
			minDist = std::min(minDist, squaredSideLength);
		}

		// Check that distance is not very small
		if (minDist < m_minContourLengthAllowed)
			continue;

		// All tests are passed. Save marker candidate:
		Marker m;

		for (int i = 0; i < 4; i++)
			m.m_points.push_back(cv::Point2f(approxCurve[i].x, approxCurve[i].y));

		// Sort the points in clockwise order
		// Trace a line between the first and second point.
		// If the third point is at the right side, then the points are anti-clockwise
		cv::Point v1 = m.m_points[1] - m.m_points[0];
		cv::Point v2 = m.m_points[2] - m.m_points[0];

		double o = (v1.x * v2.y) - (v1.y * v2.x);

		//sort points in clockwise order
		if (o < 0.0)
			std::swap(m.m_points[1], m.m_points[3]);

		possibleMarkers.push_back(m);
	}


	// Remove these elements which corners are too close to each other.  
	// First detect candidates for removal:
	std::vector< std::pair<int, int> > tooNearCandidates;
	for (size_t i = 0; i < possibleMarkers.size(); i++)
	{
		const Marker& m1 = possibleMarkers[i];

		//calculate the average distance of each corner to the nearest corner of the other marker candidate
		for (size_t j = i + 1; j < possibleMarkers.size(); j++)
		{
			const Marker& m2 = possibleMarkers[j];

			float distSquared = 0;

			for (int c = 0; c < 4; c++)
			{
				cv::Point v = m1.m_points[c] - m2.m_points[c];
				distSquared += v.dot(v);
			}

			distSquared /= 4;

			if (distSquared < 100)
			{
				tooNearCandidates.push_back(std::pair<int, int>(i, j));
			}
		}
	}

	// Mark for removal the element of the pair with smaller perimeter
	std::vector<bool> removalMask(possibleMarkers.size(), false);

	for (size_t i = 0; i < tooNearCandidates.size(); i++)
	{
		float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].m_points);
		float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].m_points);

		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;

		removalMask[removalIndex] = true;
	}

	// Return candidates
	detectedMarkers.clear();
	for (size_t i = 0; i < possibleMarkers.size(); i++)
	{
		if (!removalMask[i])
			detectedMarkers.push_back(possibleMarkers[i]);
	}
}

void MarkerDetector::recognizeMarkers(const cv::Mat& grayscale, std::vector<Marker>& detectedMarkers)
{
	std::vector<Marker> goodMarkers;

	// Identify the markers
	for (size_t i = 0; i < detectedMarkers.size(); i++)
	{
		Marker& marker = detectedMarkers[i];

		// Find the perspective transformation that brings current marker to rectangular form
		cv::Mat markerTransform = cv::getPerspectiveTransform(marker.m_points, m_markerCorners2d);

		// Transform image to get a canonical marker image
		cv::warpPerspective(grayscale, canonicalMarkerImage, markerTransform, markerSize);

#ifdef SHOW_DEBUG_IMAGES
		{
			cv::Mat markerImage = grayscale.clone();
			marker.drawContour(markerImage);
			cv::Mat markerSubImage = markerImage(cv::boundingRect(marker.m_points));

			cv::imshow("Source marker" + ToString(i), markerSubImage);
			cv::imshow("Marker " + ToString(i) + " after warp", canonicalMarkerImage);
		}
#endif

		int nRotations;
		int id = Marker::getMarkerId(canonicalMarkerImage, nRotations);
		if (id != -1)
		{
			marker.m_id = id;
			// sort the points so that they are always in the same order no matter the camera orientation
			// clockwise rotation
			std::rotate(marker.m_points.begin(), marker.m_points.begin() + 4 - nRotations, marker.m_points.end());

			goodMarkers.push_back(marker);
		}
	}

	// Refine marker corners using sub pixel accuracy
	if (goodMarkers.size() > 0)
	{
		std::vector<cv::Point2f> preciseCorners(4 * goodMarkers.size());

		for (size_t i = 0; i < goodMarkers.size(); i++)
		{
			const Marker& marker = goodMarkers[i];

			for (int c = 0; c < 4; c++)
			{
				preciseCorners[i * 4 + c] = marker.m_points[c];
			}
		}

		cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01);
		cv::cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), termCriteria);

		// Copy refined corners position back to markers
		for (size_t i = 0; i < goodMarkers.size(); i++)
		{
			Marker& marker = goodMarkers[i];

			for (int c = 0; c < 4; c++)
			{
				marker.m_points[c] = preciseCorners[i * 4 + c];
			}
		}
	}

#ifdef SHOW_DEBUG_IMAGES
	{
		cv::Mat markerCornersMat(grayscale.size(), grayscale.type());
		markerCornersMat = cv::Scalar(0);

		for (size_t i = 0; i < goodMarkers.size(); i++)
		{
			goodMarkers[i].drawContour(markerCornersMat, cv::Scalar(255));
		}

		cv::imshow("Markers refined edges", grayscale * 0.5 + markerCornersMat);
	}
#endif

	detectedMarkers = goodMarkers;
}


void MarkerDetector::estimatePosition(std::vector<Marker>& detectedMarkers)
{
	for (size_t i = 0; i < detectedMarkers.size(); i++)
	{
		Marker& m = detectedMarkers[i];

		cv::Mat Rvec;
		cv::Mat_<float> Tvec;
		cv::Mat raux, taux;
		cv::solvePnP(m_markerCorners3d, m.m_points, camMatrix, distCoeff, raux, taux);
		raux.convertTo(Rvec, CV_32F);
		taux.convertTo(Tvec, CV_32F);

		cv::Mat_<float> rotMat(3, 3);
		cv::Rodrigues(Rvec, rotMat);

		// copy to transformation matrix
		for (int row = 0; row < 3; row++)
		{
			float *trptr = m.m_transformation.ptr<float>(row);
			float *rrptr = rotMat.ptr<float>(row);
			for (int col = 0; col < 3; col++)
			{
				trptr[col] = rrptr[col];
			}
			trptr[3] = Tvec.at<float>(row, 0);
		}
	}
}
