#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 读取图像
    string imagePath = "/home/xujiake/code/task2_opencv_project/resources/test_image.jpg";
    Mat image = imread(imagePath);
    
    if (image.empty()) {
        cout << "无法加载图像!" << endl;
        return -1;
    }
    
    // 1. 图像颜色空间转换
    Mat grayImage, hsvImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    cvtColor(image, hsvImage, COLOR_BGR2HSV);
    
    // 2. 应用滤波操作
    Mat blurImage, gaussianImage;
    blur(image, blurImage, Size(5, 5));          // 均值滤波
    GaussianBlur(image, gaussianImage, Size(5, 5), 0);  // 高斯滤波
    
    // 3. 特征提取 - 提取红色区域
    // 定义HSV中的红色范围（两个区间）
    Mat redMask1, redMask2, redMask;
    inRange(hsvImage, Scalar(0, 70, 50), Scalar(10, 255, 255), redMask1);
    inRange(hsvImage, Scalar(170, 70, 50), Scalar(180, 255, 255), redMask2);
    redMask = redMask1 | redMask2;
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(redMask, redMask, MORPH_CLOSE, kernel);
    morphologyEx(redMask, redMask, MORPH_OPEN, kernel);
    
    // 寻找红色轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(redMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // 绘制红色轮廓和边界框
    Mat contourImage = image.clone();
    vector<Rect> boundRects;
    
    for (size_t i = 0; i < contours.size(); i++) {
        // 计算轮廓面积
        double area = contourArea(contours[i]);
        if (area < 100) continue;  // 过滤小面积轮廓
        
        // 绘制轮廓
        drawContours(contourImage, contours, i, Scalar(0, 255, 0), 2);
        
        // 计算边界框并绘制
        Rect rect = boundingRect(contours[i]);
        boundRects.push_back(rect);
        rectangle(contourImage, rect, Scalar(255, 0, 0), 2);
        
        // 输出轮廓面积
        cout << "轮廓 " << i << " 面积: " << area << endl;
    }
    
    // 4. 提取高亮区域并进行图形学处理
    Mat brightRegion;
    inRange(hsvImage, Scalar(0, 0, 200), Scalar(180, 30, 255), brightRegion);
    
    // 二值化
    Mat binary;
    threshold(grayImage, binary, 200, 255, THRESH_BINARY);
    
    // 膨胀和腐蚀
    Mat dilated, eroded;
    dilate(binary, dilated, kernel);
    erode(binary, eroded, kernel);
    
    // 漫水填充
    Mat floodFilled = image.clone();
    Point seedPoint(100, 100);  // 选择一个种子点
    Scalar newVal(0, 255, 255);  // 填充颜色
    Rect rect;
    floodFill(floodFilled, seedPoint, newVal, &rect, Scalar(10, 10, 10), 
              Scalar(10, 10, 10), FLOODFILL_FIXED_RANGE);
    
    // 5. 图像绘制
    Mat drawImage = image.clone();
    // 绘制圆形
    circle(drawImage, Point(100, 100), 50, Scalar(0, 0, 255), 2);
    // 绘制方形
    rectangle(drawImage, Point(200, 200), Point(300, 300), Scalar(255, 0, 0), 2);
    // 绘制文字
    putText(drawImage, "OpenCV Demo", Point(150, 400), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    
    // 6. 图像处理
    // 旋转35度
    Mat rotated;
    Point2f center(image.cols/2.0, image.rows/2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, 35, 1.0);
    warpAffine(image, rotated, rotationMatrix, image.size());
    
    // 裁剪为原图的左上角1/4
    Mat cropped = image(Rect(0, 0, image.cols/2, image.rows/2));
    
    // 显示所有结果
    imshow("Original Image", image);
    imshow("Gray Image", grayImage);
    imshow("HSV Image", hsvImage);
    imshow("Blurred Image", blurImage);
    imshow("Gaussian Blurred Image", gaussianImage);
    imshow("Red Mask", redMask);
    imshow("Contours and Bounding Boxes", contourImage);
    imshow("Bright Region", brightRegion);
    imshow("Binary", binary);
    imshow("Dilated", dilated);
    imshow("Eroded", eroded);
    imshow("Flood Filled", floodFilled);
    imshow("Drawing", drawImage);
    imshow("Rotated", rotated);
    imshow("Cropped", cropped);
    
    // 保存结果图像
    imwrite("灰度图.jpg", grayImage);
    imwrite("HSV图.jpg", hsvImage);
    imwrite("均值滤波图.jpg", blurImage);
    imwrite("高斯滤波图.jpg", gaussianImage);
    imwrite("红色区域掩码.jpg", redMask);
    imwrite("红色外轮廓和边界框.jpg", contourImage);
    imwrite("亮度图.jpg", brightRegion);
    imwrite("二值化图像.jpg", binary);
    imwrite("膨胀处理.jpg", dilated);
    imwrite("腐蚀处理.jpg", eroded);
    imwrite("漫水填充处理.jpg", floodFilled);
    imwrite("绘制图形和文字.jpg", drawImage);
    imwrite("旋转.jpg", rotated);
    imwrite("裁剪.jpg", cropped);
    
    waitKey(0);
    return 0;
}