//串行解密工程
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_functions.h>
#include <opencv2\opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include<cstring>
#include <cmath>
using namespace std;
using namespace cv;

void Getline(string &str)
{
     getline(cin, str);
     cout << "str = " << str << endl;
}
void Getbits(string str,int * code)
{
    int i = 0;
    int k = 0;
    while (str[i] != '\0')
    {
        //cout << str[i];
        for (int j = 0; j < 8; j++)
        {
            //cout << str[i]%2;
            code[k++] = str[i] % 2; //这里溢出危险没有解决
            str[i]=str[i]>>1;
        }
        i++;
    }
}
void Getline_from_bits(char str[10000], int* code)
{
    int i = 0;
    int k = 0;
    for (i = 0; i < 10000; i++)
    {
        for (int j = 0; j < 8; j++)
            str[i] += uchar(code[i*8 + j] * pow(2,j));
    }
    cout << "提取出隐藏信息:"<<endl;
    cout << str<<endl;
}
void extract(Mat img, int a[512][512])
{
    int cols = img.cols;
    int rows = img.rows;
    for (int j = 0; j < rows; j++)
        for (int i = 0; i < cols; i ++)
        {
            a[i][j] = int(img.at<uchar>(i, j));
        }
}
void dextract(Mat& img, int a[512][512])
{
    int cols = img.cols;
    int rows = img.rows;
    for (int j = 0; j < rows; j++)
        for (int i = 0; i < cols; i++)
        {
            img.at<uchar>(i, j) = uchar(a[i][j]);
        }
}
void is_same_pic(Mat img1, Mat img2, Mat img3)
{
    int flag = 0;
    int cols = img1.cols;
    int rows = img1.rows;
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < rows; i++)
        {
            if (int(img1.at<uchar>(i, j)) != int(img2.at<uchar>(i, j)))
            {
                //cout << i << " " << j << endl;
                //if (i > 0 && j > 0)
                //    cout << int(img1.at<uchar>(i - 1, j)) << " " << int(img3.at<uchar>(i - 1, j)) << " " << int(img2.at<uchar>(i - 1, j)) << endl;
                //cout << int(img1.at<uchar>(i, j)) << " " << int(img3.at<uchar>(i, j)) << " " << int(img2.at<uchar>(i, j)) << endl;
                //if (i < 511 && j < 511)
                //    cout << int(img1.at<uchar>(i + 1, j)) << " " << int(img3.at<uchar>(i + 1, j)) << " " << int(img2.at<uchar>(i + 1, j)) << endl;
                flag++;
            }
        }
    if (flag)
        cout << "两张图片不相同\n" << "不同像素点数量:" << flag << endl;
    else
        cout << "两张图片相同\n";
    return;
}
int main()
{
    //时间统计变量-串行行解密
    cudaEvent_t time_total_start, time_total_end;   //程序运行总时间
    cudaEvent_t time_decryption_start, time_decryption_end;       //加密计算map时间
    cudaEventCreate(&time_total_start); cudaEventCreate(&time_total_end);
    cudaEventCreate(&time_decryption_start); cudaEventCreate(&time_decryption_end);
    //opencv 不输出异常日志
    cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    Mat provious_img = imread("lenna_pro.bmp", 0);
    Mat img_middle = imread("lenna_en.bmp", 0);
    Mat img = imread("lenna_en.bmp", 0);
    if (img.empty())
    {
        cout << "图片读取失败！" << "\n";
        return -1;
    }//检测图像读取
    cout << "读取图像成功\n此为串行解密程序\n";
    //设置密文存放空间
    char str[10000];
    int code[512*256];                  //最多嵌入40个字符40*8=320
    memset(str, 0, sizeof(str));
    memset(code, 0, sizeof(code));
    //开始统计总时间
    cudaEventRecord(time_total_start, 0);
    //初始化图像数据
    int pic_value[512][512];        //用于实际存储图像的值
    bool map[512][256];             //实际上限是256
    bool insert_map[512][256];      //实际只能插入256个
    int h[512][256];
    int l[512][256];
    int h_p[512][256];
    memset(map, 0, sizeof(map));
    memset(insert_map, 0, sizeof(insert_map));
    extract(img, pic_value);        //获取灰度图数据
    int cols = img.cols;
    int rows = img.rows;
    int k = 0;
    //计算map与insert_map
    cudaEventRecord(time_decryption_start, 0);
    for (int j=0;j<rows;j++)
        for (int i = 0; i < cols/2; i++)//每个像素对的计算
        {
            //x指向像素对中的较大值
            int* x = (pic_value[j][i * 2] >= pic_value[j][i * 2 + 1] ? &pic_value[j][i * 2] : &pic_value[j][i * 2 + 1]);
            int* y = (pic_value[j][i * 2] >= pic_value[j][i * 2 + 1] ? &pic_value[j][i * 2 + 1] : &pic_value[j][i * 2]);
            h_p[j][i] = *x - *y;
            l[j][i] = (*x + *y) / 2;
            bool tag1 = (h_p[j][i] <= 2 * (255 - l[j][i])) && (h_p[j][i] <= (2 * l[j][i] + 1));
            if (tag1)//嵌入了数据
            {
                code[k++] = h_p[j][i]%2;//提取密文bits编码
                h[j][i] = h_p[j][i] / 2;
            }
            else
            {
                h[j][i] = h_p[j][i];
            }
            
            //图像还原
            *x = l[j][i] + (h[j][i] + 1) / 2;
            *y = l[j][i] - (h[j][i]) / 2;
        }
    cudaEventRecord(time_decryption_end, 0);
    //接下来这段只能用串行利用map与insert_map计算出code
    //从code所代表的比特流获取str字符串
    Getline_from_bits(str, code);
    //总时间统计截止
    cudaEventRecord(time_total_end, 0);
    //图像处理
    dextract(img, pic_value);          //写入img数据 
    imshow("decryed", img);            //显示图片
    waitKey(3000);
    imwrite("./lenna_de.bmp", img);     //将解密完图像写入磁盘
    waitKey(1000);
    cout << "完全串行解密完成,写入磁盘成功.";
    //对比解密还原图片与原图片失真情况
    is_same_pic(provious_img, img,img_middle);
    cout << "串行解密时间统计如下:" << endl;
    float time_total, time_decryption;
    cudaEventElapsedTime(&time_total, time_total_start, time_total_end);
    cudaEventElapsedTime(&time_decryption, time_decryption_start, time_decryption_end);
    cout << "总运行时间(不包含图像处理时间):";
    cout << time_total << endl;
    cout << "解密与重写图像像素时间:";
    cout << time_decryption << endl;
    return 0;
    
}