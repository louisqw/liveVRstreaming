#include <deque>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include<sys/time.h>

#define RTSP_URL1 "rtsp://admin:As123456@10.112.128.250:10001/"
#define RTSP_URL2 "rtsp://admin:As123456@10.112.128.250:10002/"
#define RTSP_URL3 "rtsp://admin:As123456@10.112.128.250:10003/"
#define RTSP_URL4 "rtsp://admin:As123456@10.112.128.250:10004/"
#define RTSP_URL5 "rtsp://admin:As123456@10.112.128.250:10005/"

//#define RTSP_URL1 "rtsp://admin:As123456@10.112.128.250:8005/"
//#define RTSP_URL2 "rtsp://admin:As123456@10.112.128.250:8001/"
//#define RTSP_URL3 "rtsp://admin:As123456@10.112.128.250:8002/"
//#define RTSP_URL4 "rtsp://admin:As123456@10.112.128.250:8003/"
//#define RTSP_URL5 "rtsp://admin:As123456@10.112.128.250:8004/"
#define IN_WIDTH   1920
#define IN_HEIGHT  1080
#define OUT_WIDTH  4084
#define OUT_HEIGHT 1360
#define CUT_HEIGHT 0	   //视频需要去掉的高度
#define TOP_CUT_HEIGHT 0
//#define PIPCONFIG_PATH "PIPConfig4k"CAMERA_NO"02"  //合并参数
#define BIT_RATE 6000000
#define FRAME_RATE 25
#define RTMP_URL "rtmp://10.112.213.143/live/1"
#define MAX_QUEUE_SIZE 1
#define RTSP_SOURCE_NUM 5

#define max(x, y) (x)>(y)?(x):(y)
#define min(x, y) (x)<(y)?(x):(y)




struct argument{
    int index;
    char url[255];
};

typedef struct src_img{
    uint8_t* img;
    long time;
}SRC_IMG;
