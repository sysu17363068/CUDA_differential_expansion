//串行加密工程
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_functions.h>
#include <opencv2\opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <cstring>
#include <ctime>
using namespace std;
using namespace cv;
void Getline(string &str)
{
     cout << "请输入密文" << endl;
     getline(cin, str);
     cout << "str = " << str << endl;
}
void Getbits(string str,int * code)
{
    int i = 0;
    int k = 0;
    while (str[i] != '\0')
    {
        for (int j = 0; j < 8; j++)
        {
            code[k++] = str[i] % 2; //这里溢出危险没有解决
            str[i]=str[i]>>1;  
        }
        i++;
    }
}
void extract(Mat img, int a[512][512])
{
    int cols = img.cols;
    int rows = img.rows;
    for (int j = 0; j < rows; j++)
        for (int i = 0; i < cols; i++)
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
int main()
{
    //时间统计变量-串行加密
    cudaEvent_t time_total_start, time_total_end;   //程序运行总时间
    cudaEvent_t time_map_start, time_map_end;       //加密计算map时间
    cudaEvent_t time_pic_start, time_pic_end;       //给图像嵌密消耗时间
    cudaEventCreate(&time_total_start);cudaEventCreate(&time_total_end);
    cudaEventCreate(&time_map_start); cudaEventCreate(&time_map_end);
    cudaEventCreate(&time_pic_start); cudaEventCreate(&time_pic_end);
    //opencv 不输出异常日志
    cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    Mat img = imread("lenna_pro.bmp", 0);
    if (img.empty())
    {
        cout << "图片读取失败！" << "\n";
        return -1;
    }//检测图像读取
    cout << "读取图像成功\n此为串行加密程序\n";
    //获取输入密文并设置密文存放空间
    string str;
    int code[512*256];                  //最多嵌入40个字符40*8=320 400是为了防止溢出设计的
    memset(code, 0, sizeof(code));
    Getline(str);
    Getbits(str, code);
    //从此开始统计运行时间
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
    cudaEventRecord(time_map_start, 0);
    for (int j=0;j<rows;j++)
        for (int i = 0; i < cols/2; i++)//每个像素对的计算
        {
            //x指向像素对中较大值的位置
            int* x = (pic_value[j][i*2] >= pic_value[j][i*2+1] ? &pic_value[j][i*2] : &pic_value[j][i * 2 + 1]);
            int* y = (pic_value[j][i*2] >= pic_value[j][i*2+1] ? &pic_value[j][i*2+1] : &pic_value[j][i * 2]);
            //差分运算
            h[j][i] = *x - *y;
            l[j][i] = (*x + *y) / 2;
            h_p[j][i] = 2 * h[j][i] + 1;
            //确保嵌入后不会溢出
            bool tag1 = (h_p[j][i] <= 2 * (255 - l[j][i])) && (h_p[j][i] <= (2 * l[j][i] + 1));
            //如果嵌入后溢出，但是嵌入前不溢出则直接修改最后一位确保解密正常完成
            bool tag2 = (h[j][i] <= 2 * (255 - l[j][i])) && (h[j][i] <= (2 * l[j][i] + 1));
            if (tag1)   //扩展后没有溢出
            {
                h_p[j][i] = 2 * h[j][i] + int(code[k++]);
            }
            else if (tag2) //扩展前没有溢出
            {
                h_p[j][i] = (h[j][i]-h[j][i]%2) + int(code[k++]);
            }
            else
            {
                h_p[j][i] = h[j][i];//不进行像素修改
            }
            *x = l[j][i] + (h_p[j][i] + 1) / 2;
            *y = l[j][i] - (h_p[j][i]) / 2 ;
        }
    cudaEventRecord(time_map_end, 0);    
    //将密文嵌入至图像数据之中
    //总时间截止
    cudaEventRecord(time_total_end, 0);
    //图像处理
    dextract(img, pic_value);       //写入img数据
    imshow("encryption", img);          //显示图片
    waitKey(3000);                  
    imwrite("./lenna_en.bmp", img); //将加密后图像写入磁盘
    waitKey(1000);
    cout << "加密完成,写入磁盘成功.";
    cout << "成功写入字节数:" << (k/8)<< endl;
    cout << "串行加密时间统计如下:" << endl;
    float time_total, time_map, time_pic;
    cudaEventElapsedTime(&time_total, time_total_start, time_total_end);
    cudaEventElapsedTime(&time_map, time_map_start, time_map_end);
    cout << "总运行时间(不包含图像处理时间):";
    cout << time_total << endl;
    cout << "计算map时间:";
    cout << time_map  << endl;
    return 0;
    
}