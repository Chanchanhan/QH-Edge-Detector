
namespace tk {

class View {
public:
	float* GetKMat() { return &k_ary[9*scale]; }

	int width() { return width_ / (1 << scale); }
	int height() { return height_ / (1 << scale); }

	template<typename T>
	bool PointInFrame(T x, T y) {
		if (x >= 0 && x < width() && y >= 0 && y < height()) return true;
		else return false;
	}

	bool PointInFrame(cv::Point& point) {
		return PointInFrame(point.x, point.y);
	}

	int width_, height_;
	cv::Mat k_mat[3];
	float k_ary[9*3];
	int scale;
};

} // namespace tk