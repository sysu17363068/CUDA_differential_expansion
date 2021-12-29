//CUDA并行解密工程
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
 __global__ void encryption(int* pic_value_dev, bool* map_dev, bool* insert_map_dev,int* h_dev, int* l_dev, int* h_p_dev)
 {
     //threadIdx.x 为x坐标256 blockIdx.x为y坐标 按512分配
     const int j = blockIdx.x;
     const int i = threadIdx.x;
     int* x = (pic_value_dev[i * 2 + 512 * j] >= pic_value_dev[i * 2 + 1 + 512 * j] ? &pic_value_dev[i * 2 + 512 * j] : &pic_value_dev[i * 2 + 1 + 512 * j]);
     int* y = (pic_value_dev[i * 2 + 512 * j] >= pic_value_dev[i * 2 + 1 + 512 * j] ? &pic_value_dev[i * 2 + 1 + 512 * j] : &pic_value_dev[i * 2 + 512 * j]);
     h_p_dev[i + 256 * j] = *x - *y;
     l_dev[i + 256 * j] = (*x + *y) / 2;
     h_dev[i + 256 * j] = h_p_dev[i + 256 * j]/2;
     //三目运算等价分支运算
     //if ((h_p_dev[i + 256 * j] <= 2 * (255 - l_dev[i + 256 * j])) && (h_p_dev[i + 256 * j] <= (2 * l_dev[i + 256 * j] + 1)))
     //{
     //   map_dev[i + 256 * j] = 1;
     //   insert_map_dev[i + 256 * j] = (h_p_dev[i + 256 * j] % 2);
     //   *x = l_dev[i + 256 * j] + (h_dev[i + 256 * j] + 1) / 2;
     //   *y = l_dev[i + 256 * j] + (h_dev[i + 256 * j]) / 2;
     //}
     //else
     //{
     //   map_dev[i + 256 * j] = 0;
     //   *x = *x;
     //   *y = *y;
     //}
     //三目运算
     map_dev[i + 256 * j] = ((h_p_dev[i + 256 * j] <= 2 * (255 - l_dev[i + 256 * j])) && (h_p_dev[i + 256 * j] <= (2 * l_dev[i + 256 * j] + 1)));//得到了map
     insert_map_dev[i + 256 * j] = map_dev[i + 256 * j] * (h_p_dev[i + 256 * j] % 2);
     *x = (map_dev[i + 256 * j] ? l_dev[i + 256 * j] + (h_dev[i + 256 * j] + 1) / 2 : *x);
     *y = (map_dev[i + 256 * j] ? l_dev[i + 256 * j] - (h_dev[i + 256 * j]) / 2 : *y);
 }
void check(cudaError_t error)
 {
     if (error != cudaSuccess)
     {
         printf("ERROR: %s:%d,", __FILE__, __LINE__);
         printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
         exit(1);
     }
     //else
     //    printf("success\n");
 }
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
void Getline_from_bits(char str[1000], int* code)
{
    int i = 0;
    int k = 0;
    for (i = 0; i < 1000; i++)
    {
        for (int j = 0; j < 8; j++)
            str[i] += uchar(code[i * 8 + j] * pow(2, j));
    }
    cout << "提取出隐藏信息:" << endl;
    cout << str << endl;
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
void is_same_pic(Mat img1, Mat img2,Mat img3)
{
    int flag = 0;
    int cols = img1.cols;
    int rows = img1.rows;
    for(int j=0;j<cols;j++)
        for (int i = 0; i < rows; i++)
        {
            if (int(img1.at<uchar>(i, j)) != int(img2.at<uchar>(i, j)))
            {
                //cout << i << " " << j<<endl;
                //if (i > 0 && j > 0)
                //    cout << int(img1.at<uchar>(i-1, j)) << " " << int(img3.at<uchar>(i-1, j)) << " " << int(img2.at<uchar>(i-1, j)) << endl;
                //cout << int(img1.at<uchar>(i, j))<<" "<< int(img3.at<uchar>(i, j)) << " " << int(img2.at<uchar>(i, j)) << endl;
                //if (i<511&&j<511)
                //    cout << int(img1.at<uchar>(i+1, j)) << " " << int(img3.at<uchar>(i+1, j)) << " " << int(img2.at<uchar>(i+1, j)) << endl;
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
    //时间统计变量-并行解密
    cudaEvent_t time_total_start, time_total_end;   //程序运行总时间
    cudaEvent_t time_decryption_start, time_decryption_end;       //加密计算map时间
    cudaEventCreate(&time_total_start); cudaEventCreate(&time_total_end);
    cudaEventCreate(&time_decryption_start); cudaEventCreate(&time_decryption_end);
    //opencv不再异常输出日志
    cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    //图像数据读取
    Mat provious_img = imread("lenna_pro.bmp", 0);
    Mat img_middle = imread("lenna_en.bmp", 0);
    Mat img = imread("lenna_en.bmp", 0);
    if (img.empty())
    {
        std::cout << "图片读取失败！" << "\n";
        return -1;
    }//检测图像读取
    cout << "读取图像成功\n此为CUDA并行解密程序\n";
    //密文编码存放空间
    char str[1000];
    int code[512*256];                      //最多嵌入50个字符20*8=160
    memset(str, 0, sizeof(str));
    memset(code, 0, sizeof(code));
    //开始统计总时间
    cudaEventRecord(time_total_start, 0);
    //图像数据定义部分与初始化
    int pic_value[512][512];            //用于实际存储图像的值
    extract(img, pic_value);            //获取灰度图数据
    bool map[512][256];                 //map实际上限是256
    bool insert_map[512][256];          //插入 0/1 bit的map
    int h[512][256];                    //原图差值
    int l[512][256];                    //原图平均值
    int h_p[512][256];                  //加密图差值
    int* h_dev = NULL;                 //各项数据的device拷贝
    int* l_dev = NULL;
    int* h_p_dev = NULL;
    bool* insert_map_dev = NULL;        //显存insert_map
    bool* map_dev = NULL;               //device map
    int* pic_value_dev = NULL;          //显存图像数据
    memset(map, 0, sizeof(map));
    memset(insert_map, 0, sizeof(insert_map));
    check(cudaMalloc((void**)&h_dev, sizeof(int) * 512 * 256));
    check(cudaMalloc((void**)&l_dev, sizeof(int) * 512 * 256));
    check(cudaMalloc((void**)&h_p_dev, sizeof(int) * 512 * 256));
    check(cudaMalloc((void**)&pic_value_dev, sizeof(pic_value)));
    check(cudaMemcpy(pic_value_dev, pic_value, 512 * 512 * sizeof(int), cudaMemcpyHostToDevice));
    check(cudaMalloc((void**)&map_dev, sizeof(map)));
    check(cudaMemcpy(map_dev, map, 512 * 256 * sizeof(bool), cudaMemcpyHostToDevice));
    check(cudaMalloc((void**)&insert_map_dev, sizeof(insert_map)));
    check(cudaMemcpy(insert_map_dev, insert_map, 512 * 256 * sizeof(bool), cudaMemcpyHostToDevice));
    
    int cols = img.cols;
    int rows = img.rows;
    //计算map与insert_map
    cudaEventRecord(time_decryption_start, 0);
    encryption<<<512,256>>>(pic_value_dev, map_dev, insert_map_dev, h_dev, l_dev, h_p_dev);
    cudaEventRecord(time_decryption_end, 0);
    //map , insert_map 与图像数据 拷贝至 host
    check(cudaMemcpy(map, map_dev, 512 * 256 * sizeof(bool), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(insert_map, insert_map_dev, 512 * 256 * sizeof(bool), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(pic_value, pic_value_dev, 512 * 512 * sizeof(int), cudaMemcpyDeviceToHost));
    //接下来这段只能用顺行利用map与insert_map计算出code
    int k = 0;
    for (int j = 0; j < rows; j++)
    {
        for (int i = 0; i < cols / 2; i++)
        {
            if (map[j][i])
            {
                code[k++] = int(insert_map[j][i]);//提取编码
            }
        }
    }
    //从code所代表的比特流获取隐藏字符串
    Getline_from_bits(str, code);
    //总时间统计截止
    cudaEventRecord(time_total_end, 0);
    //图像处理
    dextract(img, pic_value);
    imshow("decryed", img);
    waitKey(3000);
    imwrite("./lenna_de.bmp", img);
    waitKey(1000);
    cout << "并行解密完成,写入磁盘成功.";
    //查看图像失真情况
    is_same_pic(provious_img, img,img_middle);
    cout << "CUDA并行解密时间统计如下:" << endl;
    float time_total, time_decryption;
    cudaEventElapsedTime(&time_total, time_total_start, time_total_end);
    cudaEventElapsedTime(&time_decryption, time_decryption_start, time_decryption_end);
    cout << "总运行时间(不包含图像处理时间):";
    cout << time_total << endl;
    cout << "解密与重写图像像素时间:";
    cout << time_decryption << endl;
    //释放显存
    cudaFree(pic_value_dev);
    cudaFree(map_dev);
    cudaFree(insert_map_dev);
    cudaFree(h_dev);
    cudaFree(h_p_dev);
    cudaFree(l_dev);
    return 0;
    
}