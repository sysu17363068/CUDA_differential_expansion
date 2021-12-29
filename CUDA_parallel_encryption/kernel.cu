//CUDA并行加密工程
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_functions.h>
#include <opencv2\opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <cstring>
#include <ctime>
using namespace std;
using namespace cv;
__global__ void acc_map(int* pic_value_dev, bool* map_dev, int* h_dev, int* l_dev, int* h_p_dev)
{
    //threadIdx.x 为x坐标256 blockIdx.x为y坐标 按512分配
    const int j = blockIdx.x;
    const int i = threadIdx.x;
    int* x = (pic_value_dev[i * 2 + 512 * j] >= pic_value_dev[i * 2 + 1 + 512 * j] ? &pic_value_dev[i * 2 + 512 * j] : &pic_value_dev[i * 2 + 1 + 512 * j]);
    int* y = (pic_value_dev[i * 2 + 512 * j] >= pic_value_dev[i * 2 + 1 + 512 * j] ? &pic_value_dev[i * 2 + 1 + 512 * j] : &pic_value_dev[i * 2 + 512 * j]);
    h_dev[i + 256 * j] = *x - *y;
    l_dev[i + 256 * j] = (*x + *y) / 2;
    h_p_dev[i + 256 * j] = 2 * h_dev[i + 256 * j] +1;
    bool flag1 = (h_p_dev[i + 256 * j] <= 2 * (255 - l_dev[i + 256 * j]))&& (h_p_dev[i + 256 * j] <= (2 * l_dev[i + 256 * j] + 1));
    bool flag2 = (h_dev[i + 256 * j] <= 2 * (255 - l_dev[i + 256 * j])) && (h_dev[i + 256 * j] <= (2 * l_dev[i + 256 * j] + 1));
    //三目运算
    map_dev[i + 256 * j] = flag1||flag2;
    h_p_dev[i + 256 * j]= flag1*(h_p_dev[i + 256 * j] -1);   //两倍的h
    h_p_dev[i + 256 * j]= (h_p_dev[i + 256 * j] ? (h_p_dev[i + 256 * j]):flag2*h_dev[i + 256 * j]);   //一倍的h去掉最低为位
    //三目运算等价分支运算
    //if (flag1)
    //{
    //    h_p_dev[i + 256 * j] = h_dev[i + 256 * j] * 2;
    //    map_dev[i + 256 * j] = true;
    //}
    //else if(flag2)
    //{
    //    h_p_dev[i + 256 * j] = h_dev[i + 256 * j]/2*2;//最后一位置0
    //    map_dev[i + 256 * j] = true;
    //}
}
__global__ void encrypt_pic(int* pic_value_dev, bool* map_dev,bool * insert_map_dev, int* h_dev, int* l_dev, int* h_p_dev)
{
    const int j = blockIdx.x;
    const int i = threadIdx.x;
    //X指向像素对中较大的值
    int* x = (pic_value_dev[i * 2 + 512 * j] >= pic_value_dev[i * 2 + 1 + 512 * j] ? &pic_value_dev[i * 2 + 512 * j] : &pic_value_dev[i * 2 + 1 + 512 * j]);
    int* y = (pic_value_dev[i * 2 + 512 * j] >= pic_value_dev[i * 2 + 1 + 512 * j] ? &pic_value_dev[i * 2 + 1 + 512 * j] : &pic_value_dev[i * 2 + 512 * j]);
    //三目运算
    h_p_dev[i + 256 * j] = h_p_dev[i + 256 * j] + int(insert_map_dev[i + 256 * j]);
    *x = (map_dev[i + 256 * j] ? l_dev[i + 256 * j] + (h_p_dev[i + 256 * j] + 1) / 2 : *x);
    *y = (map_dev[i + 256 * j] ? l_dev[i + 256 * j] - (h_p_dev[i + 256 * j]) / 2 : *y);
    //等价分支运算
//if (map_dev[i + 256 * j])
//{
//    if (insert_map_dev[i + 256 * j])
//        h_p_dev[i + 256 * j] = h_dev[i + 256 * j] * 2 + 1;
//    else
//        h_p_dev[i + 256 * j] = h_dev[i + 256 * j] * 2;
//}
//else
//    h_p_dev[i + 256 * j] = h_dev[i + 256 * j];
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
     cout << "请输入密文"<<endl;
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
            //cout << str[i]%2;
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
    //时间统计变量-CUDA并行加密
    cudaEvent_t time_total_start, time_total_end;   //程序运行总时间
    cudaEvent_t time_map_start, time_map_end;       //加密计算map时间
    cudaEvent_t time_pic_start, time_pic_end;       //给图像嵌密消耗时间
    cudaEventCreate(&time_total_start); cudaEventCreate(&time_total_end);
    cudaEventCreate(&time_map_start); cudaEventCreate(&time_map_end);
    cudaEventCreate(&time_pic_start); cudaEventCreate(&time_pic_end);
    //opencv不再异常输出日志
    cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    //图片读取部分
    Mat img = imread("lenna_pro.bmp", 0);
    if (img.empty())
    {
        cout << "图片读取失败!" << "\n";
        return -1;
    }//检测图像读取
    cout << "图片读取成功!" << endl<<"此为CUDA并行加密程序!"<<endl;
    //密文编码处理部分
    string str;
    int code[512*256];                      //最多嵌入50个字符20*8=160 
    memset(code, 0, sizeof(code));  
    Getline(str);                       //获取键盘输入的ASCii字符串
    Getbits(str, code);                 //将str字符串编码为二进制存放在code数组中
    //从此开始统计运行时间
    cudaEventRecord(time_total_start, 0);
    //图像数据定义部分与初始化
    int pic_value[512][512];            //用于实际存储图像的值
    extract(img, pic_value);            //获取灰度图数据
    bool map[512][256];                 //map实际上限是256
    bool insert_map[512][256];          //插入 0/1 bit的map
    int h[512][256];                    //原图差值
    int l[512][256];                    //原图平均值
    int h_p[512][256];                  //加密图差值
    int * h_dev = NULL;                 //各项数据的device拷贝
    int * l_dev = NULL;
    int * h_p_dev = NULL;
    bool* insert_map_dev = NULL;        //显存insert_map
    bool* map_dev = NULL;               //device map
    int *pic_value_dev = NULL;          //显存图像数据
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
    //图像边长获取，程序目前只能使用512*512灰度图像
    int cols = img.cols;
    int rows = img.rows;
    cudaError_t result;
    //计算出map
    cudaEventRecord(time_map_start, 0);
    acc_map << <512, 256 >> > (pic_value_dev,map_dev,h_dev,l_dev,h_p_dev);
    cudaEventRecord(time_map_end, 0);
    //map 拷贝至 device
    check(cudaMemcpy(map, map_dev, 512 * 256 * sizeof(bool), cudaMemcpyDeviceToHost));
    //接下来这段只能用顺行计算出insert_map
    int k = 0;
    for (int j = 0; j < rows; j++)
    {
        for (int i = 0; i < cols/2; i++)
        {
            if (map[j][i])
            {
                insert_map[j][i] = bool(code[k++]);//这个像素对进行嵌入
            }             
        }
    }
    //insert_map 拷贝至 device 
    check(cudaMemcpy(insert_map_dev, insert_map, 512 * 256 * sizeof(bool), cudaMemcpyHostToDevice));
    //使用并行将密文嵌入至图像数据之中
    cudaEventRecord(time_pic_start, 0);
    encrypt_pic<<<512,256>>> (pic_value_dev, map_dev,insert_map_dev, h_dev, l_dev, h_p_dev);
    cudaEventRecord(time_pic_end, 0);
    //pic_value copy到主存
    check(cudaMemcpy(pic_value, pic_value_dev, 512 * 512 * sizeof(int), cudaMemcpyDeviceToHost));
    //总时间截止
    cudaEventRecord(time_total_end, 0); 
    //写入img,显示,写入磁盘文件
    dextract(img, pic_value);
    imshow("lalala", img);
    waitKey(3000);
    imwrite("./lenna_en.bmp", img);
    waitKey(1000);
    cout << "加密完成,写入磁盘成功.";
    cout << "CUDA并行加密时间统计如下:" << endl;
    float time_total, time_map, time_pic;
    cudaEventElapsedTime(&time_total, time_total_start, time_total_end);
    cudaEventElapsedTime(&time_map, time_map_start, time_map_end);
    cudaEventElapsedTime(&time_pic, time_pic_start, time_pic_end);
    cout << "总运行时间(不包含图像处理时间):";
    cout << time_total << endl;
    cout << "计算map时间:";
    cout << time_map << endl;
    cout << "改写图片数据嵌密时间:";
    cout << time_pic << endl;
    //释放显存
    cudaFree(pic_value_dev);
    cudaFree(map_dev);
    cudaFree(insert_map_dev);
    cudaFree(h_dev);
    cudaFree(h_p_dev);
    cudaFree(l_dev);
    return 0;
}