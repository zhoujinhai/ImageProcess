void converityDefect(cv::Mat & projectImg){
	int w = projectImg.cols;
	memcpy(projectImg.data, imgCopy.GetData(), imgCopy.GetWidth() * imgCopy.GetHeight());
	/*cv::Mat thresh;
	cv::threshold(projectImg, thresh, 0, 255, cv::THRESH_BINARY);*/
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::morphologyEx(projectImg, projectImg, cv::MORPH_CLOSE, kernel);
	std::vector< std::vector< cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(projectImg, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	std::vector< cv::Point>cnt = contours[0];
	for (int i = 1; i < contours.size(); ++i) {
		if (contours[i].size() > cnt.size()) {
			cnt = contours[i];
		}
	}
	std::vector<int> hull;
	cv::convexHull(cnt, hull, false);
	std::vector<cv::Vec4i>  defects(cnt.size());
	cv::convexityDefects(cnt, hull, defects);

	double maxDist = 0;
	core::Vector2i point1, point2;
	for (int i = 0; i < defects.size(); ++i) {
		cv::Vec4i defect = defects[i];
		int startIdx = defect[0];
		int endIdx = defect[1];
		int farIdx = defect[2];
		int depth = defect[3];
		int sx = cnt[startIdx].x;
		int sy = cnt[startIdx].y;
		int ex = cnt[endIdx].x;
		int ey = cnt[endIdx].y;
		if (depth > maxDist) {
			maxDist = depth;
			point1 = core::Vector2i(sx, sy);
			point2 = core::Vector2i(ex, ey);
		}
	
	}
}