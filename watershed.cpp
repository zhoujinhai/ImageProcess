void watershed(cv::Mat & projectImg){
	cv::Mat projImg3;
	cv::cvtColor(projectImg, projImg3, cv::COLOR_GRAY2RGB);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(projectImg, projectImg, cv::MORPH_OPEN, kernel);
	// ---1.生成种子区域（前景）
	cv::Mat distImg;
	cv::distanceTransform(projectImg, distImg, cv::DIST_L2, cv::DIST_MASK_PRECISE);
	cv::normalize(distImg, distImg, 255, 0, cv::NORM_MINMAX, CV_8U);
	double minVal = 0.0, maxVal = 0.0;
	cv::minMaxIdx(distImg, &minVal, &maxVal);
	
	double ratio = 0.70;
	cv::Mat sureFGImg;
	cv::threshold(distImg, sureFGImg, ratio * maxVal, 255, cv::THRESH_BINARY);
	std::vector< std::vector< cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(sureFGImg, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	int cntNum = contours.size();
	double cntRatio = ratio;
	double ratioStep = 0.05;
	while (ratio >= 0.25) {
		if (cntNum >= 12) {
			break;
		}
		ratio -= ratioStep;
		cv::threshold(distImg, sureFGImg, ratio * maxVal, 255, cv::THRESH_BINARY);
		cv::findContours(sureFGImg, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
		if (contours.size() > cntNum) {
			cntNum = contours.size();
			cntRatio = ratio;
		}
	}
	cv::threshold(distImg, sureFGImg, cntRatio * maxVal, 255, cv::THRESH_BINARY);
	// ---2.查找未知区域
	sureFGImg.convertTo(sureFGImg, CV_8U);
	cv::Mat sureBGImg, unknownRegionImg;
	cv::dilate(projectImg, sureBGImg, kernel);
	cv::subtract(sureBGImg, sureFGImg, unknownRegionImg);
#ifdef _DEBUG
	if (false)
	{
		cv::imshow("unknown", unknownRegionImg);
		cv::waitKey(0);
	}
#endif
	// ---3.查找标记区域
	cv::Mat markers(sureFGImg.size(), CV_32S);
	cv::connectedComponents(sureFGImg, markers);
	// Add one to all labels so that sure background is not 0, but 1
	markers = markers + 1;
	// Now, mark the region of unknown with zero
	for (int row = 0; row < unknownRegionImg.rows; ++row) {
		for (int col = 0; col < unknownRegionImg.cols; ++col) {
			if ((int)unknownRegionImg.at<uchar>(row, col) == 255) {
				markers.at<int>(row, col) = 0;
			}
		}
	}

	// ---4. 分水岭算法
	cv::watershed(projImg3, markers);
	for (int row = 0; row < markers.rows; ++row) {
		for (int col = 0; col < markers.cols; ++col) {
			if (markers.at<int>(row, col) == -1) {
				projImg3.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
			}
		}
	}
#ifdef _DEBUG
  if (false) {
	cv::imshow("img", projImg3);
	cv::waitKey(0);
  }
#endif
}