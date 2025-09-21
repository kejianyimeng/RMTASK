#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

int main(){
    
    string image_path2="/home/xujiake/code/task2_opencv_project/resources/test_image_2.jpg";
    Mat image=imread(image_path2);
    
    if(image.empty()){
        cout<<"无法加载图像!"<<endl;
        return -1;
    }

    //高斯滤波
    Mat blur_image;
    GaussianBlur(image,blur_image,Size(5,5),0);
    //转为灰度图
    Mat gray_image;
    cvtColor(blur_image,gray_image,COLOR_BGR2GRAY);
    //二值化
    Mat binary_image;
    threshold(gray_image,binary_image,220,255,THRESH_BINARY);
    //膨胀
    Mat dilate_image;
    Mat kernel=getStructuringElement(MORPH_RECT,Size(5,5));
    dilate(binary_image,dilate_image,kernel);
    //查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(dilate_image,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    //筛选轮廓并绘制边界框
    Mat contour_image=image.clone();
    for(size_t i=0;i<contours.size();i++){
        double area=contourArea(contours[i]);
        if(area<1200 || area>1450) continue; 
        //绘制轮廓
        drawContours(contour_image,contours,(int)i,Scalar(0,255,0),2);
        //计算边界框并绘制
        Rect rect=boundingRect(contours[i]);
        rectangle(contour_image,rect,Scalar(255,0,0),2);
    }

    imshow("Contour Image",contour_image);

    imwrite("装甲板.jpg",contour_image);
    
    waitKey(0);
    return 0;
}
