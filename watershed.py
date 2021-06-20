import cv2
import numpy as np

def findBestSeed(dist):
    ratio = 0.70
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
    print("maxVal: ", maxVal, "minVal: ", minVal)
    _, sure_fg = cv2.threshold(dist, ratio * maxVal, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    contours, _ = cv2.findContours(sure_fg, 2, 2)
    cnt_num = len(contours)
    cnt_ratio = ratio
    step = 0.05
    # 根据具体情况进行设置，实现自动调节
    while ratio >= 0.25:
        if cnt_num >= 12:
            break
        ratio -= step
        _, sure_fg = cv2.threshold(dist, ratio * maxVal, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        contours, _ = cv2.findContours(sure_fg, 2, 2)
        if len(contours) > cnt_num:
            cnt_num = len(contours)
            cnt_ratio = ratio
    _, sure_fg = cv2.threshold(dist, cnt_ratio * maxVal, 255, cv2.THRESH_BINARY)
    print("cnt_ratio", cnt_ratio)
    return sure_fg


def watershed(img_path):

    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 127, 255, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 生成种子区域
    dist_img = cv2.distanceTransform(thresh, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist2 = cv2.normalize(dist_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    heat_img = cv2.applyColorMap(dist2, cv2.COLORMAP_JET)  # 生成热力图
    cv2.imshow("heat", heat_img)
    sure_fg = findBestSeed(dist2)  # sure_fg.convertTo(sure_fg, CV_8U);
    contours, hierarchy = cv2.findContours(sure_fg, 2, 2)
    print("mark: ", len(contours))

    # 查找未知区域
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(thresh, kernel)
    unknown = cv2.subtract(sure_bg, sure_fg)


    # 查找标记区域
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # 分水岭算法
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 0, 0]
    return img


if __name__ == "__main__":
    img_path = r'E:\test.png'
    img = watershed(img_path)

    cv2.waitKey(0)