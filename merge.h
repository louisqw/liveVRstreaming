//--------------------OpenCV头文件---------------
#include <opencv2/opencv.hpp>
#include <opencv2/core/version.hpp>

using namespace cv;
#if CV_VERSION_EPOCH == 2
#define OPENCV2
#include <opencv2/gpu/gpu.hpp>
using namespace cv::gpu;

#elif CV_VERSION_MAJOR == 3
#define  OPENCV3
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;
#else
#error Not support this OpenCV version
#endif


void mapping_init(int num, int*** mapGx, int*** mapGy);
void mapping(GpuMat& gpuMat,GpuMat& gpuMatSrc,int** mapGx,int** mapGy);
void tmapping_init(int num, int*** mapGx, int*** mapGy);
void YUV123(GpuMat& gpuMat, GpuMat& gpuMatSrc);
void YUV321(GpuMat& gpuMat, GpuMat& gpuMatSrc);
void GPUinit(int num);//3.28

class cvGPUinit{
	int dev;
public:
	cvGPUinit(){
		//initialize CUDA device
		int num_devices = getCudaEnabledDeviceCount();
		std::cout<<num_devices<<" devices in total"<<std::endl;
		if (num_devices <= 0) {
			std::cerr << "There is no device." << std::endl;
			return ;
		}
		int dev = -1;
		for (int i = 0; i < num_devices; i++) {
			DeviceInfo dev_info(i);
			if (dev_info.isCompatible()) {
				dev = i;
				break;
			}
		}
		if (dev < 0) {
			std::cerr << "GPU module isn't built for GPU." << std::endl;
			return ;
		}
		setDevice(dev);
		GPUinit(dev);
		std::cout<<"using GPU "<<dev<<std::endl;
	}
	cvGPUinit(int n):dev(n){
		//initialize CUDA device
		int num_devices = getCudaEnabledDeviceCount();
		std::cout<<num_devices<<" devices in total"<< std::endl;
		if (num_devices <= 0) {
			std::cerr << "There is no device." << std::endl;
			return ;
		}
		DeviceInfo dev_info(dev);
		if (!dev_info.isCompatible()) {
			std::cerr << "GPU module "<<dev<<" isn't available." << std::endl;
			return ;
		}
		setDevice(dev);
		GPUinit(dev);
		std::cout<<"using GPU "<<dev<<std::endl;
	}
};

//金字塔构建层
struct layer{
    cuda::GpuMat img_src;
    cuda::GpuMat p0;
    cuda::GpuMat p1;
    cuda::GpuMat p2;
    cuda::GpuMat p3;
    cuda::GpuMat p4;
    cuda::GpuMat p5;
    int tag;
    layer() {}
    layer(int row, int col, int num) { img_src.create(row, col, CV_8UC3); p0.create(row, col, CV_32FC3); p1.create((int)ceil(row / 2.0), (int)ceil(col / 2.0), CV_32FC3); p2.create((int)ceil(ceil(row / 2.0) / 2.0), (int)ceil(ceil(col / 2.0) / 2.0), CV_32FC3); p3.create((int)ceil(ceil(ceil(row / 2.0) / 2.0) / 2.0), (int)ceil(ceil(ceil(col / 2.0) / 2.0) / 2.0), CV_32FC3); p4.create((int)ceil(ceil(ceil(ceil(row / 2.0) / 2.0) / 2.0)/2.0), (int)ceil(ceil(ceil(ceil(col / 2.0) / 2.0) / 2.0)/2.0), CV_32FC3); p5.create((int)ceil(ceil(ceil(ceil(ceil(row / 2.0) / 2.0) / 2.0)/2.0) / 2.0), (int)ceil(ceil(ceil(ceil(ceil(col / 2.0) / 2.0) / 2.0)/2.0) / 2.0), CV_32FC3); tag = num; }
};

//金字塔构建辅助空间
struct layer_support{
    cuda::GpuMat s0;
    cuda::GpuMat s1;
    cuda::GpuMat s2;
    cuda::GpuMat s3;
    cuda::GpuMat s4;
    cuda::GpuMat s5;
    layer_support(int row, int col) { s0.create(row, col, CV_32FC3); s1.create((int)ceil(row / 2.0), (int)ceil(col / 2.0), CV_32FC3); s2.create((int)ceil(ceil(row / 2.0) / 2.0), (int)ceil(ceil(col / 2.0) / 2.0), CV_32FC3); s3.create((int)ceil(ceil(ceil(row / 2.0) / 2.0) / 2.0), (int)ceil(ceil(ceil(col / 2.0) / 2.0) / 2.0), CV_32FC3); s4.create((int)ceil(ceil(ceil(ceil(row / 2.0) / 2.0) / 2.0)/2.0), (int)ceil(ceil(ceil(ceil(col / 2.0) / 2.0) / 2.0)/2.0), CV_32FC3); s5.create((int)ceil(ceil(ceil(ceil(ceil(row / 2.0) / 2.0) / 2.0)/2.0) / 2.0), (int)ceil(ceil(ceil(ceil(ceil(col / 2.0) / 2.0) / 2.0)/2.0) / 2.0), CV_32FC3);}
};

//融合辅助空间
struct merge_support{
	cuda::GpuMat mc0;
	cuda::GpuMat mc1;
	cuda::GpuMat mc2;
	cuda::GpuMat mc3;
	cuda::GpuMat mc4;
	cuda::GpuMat mc5;
    cuda::GpuMat ms0;
    cuda::GpuMat ms1;
    cuda::GpuMat ms2;
    cuda::GpuMat ms3;
    cuda::GpuMat ms4;
    cuda::GpuMat ms5;
    cuda::GpuMat ms6;
    cuda::GpuMat ms7;
    cuda::GpuMat ms8;
    cuda::GpuMat ms9;
    merge_support(){}
    merge_support(int row1, int col1, int row2, int col2, int row3, int col3, int row, int col){
   	    mc0.create(row1, col1, CV_32FC3);
	    mc1.create(row2, col2, CV_32FC3);
	    mc2.create(row2, col2, CV_32FC3);
	    mc3.create(row2, col2, CV_32FC3);
	    mc4.create(row1, col1, CV_32FC3);
	    mc5.create(row3, col3, CV_32FC3);
        ms0.create(row, col, CV_32FC3);
        ms1.create(row, col, CV_32FC3);
        ms2.create(row, col, CV_32FC3);
        ms3.create(row, col, CV_32FC3);
        ms4.create(row, col, CV_32FC3);
        ms5.create(row, col, CV_32FC3);
        ms6.create(row, col, CV_32FC3);
        ms7.create(row, col, CV_32FC3);
        ms8.create(row, col, CV_32FC3);
        ms9.create(row, col, CV_32FC3);
    }
};

//全景帧重构空间
struct merge_all_support {
	cuda::GpuMat ma0;
	cuda::GpuMat ma1;
	cuda::GpuMat ma2;
	cuda::GpuMat ma3;
	cuda::GpuMat ma4;
	cuda::GpuMat ma5;
	cuda::GpuMat ma6;
	cuda::GpuMat ma7;
	cuda::GpuMat ma8;
    cuda::GpuMat ma9;
	merge_all_support() {}
	merge_all_support(int row, int col) {
		ma0.create(row, col, CV_32FC3);
		ma1.create(row, col, CV_32FC3);
		ma2.create(row, col, CV_32FC3);
		ma3.create(row, col, CV_32FC3);
		ma4.create(row, col, CV_32FC3);
		ma5.create(row, col, CV_32FC3);
		ma6.create(row, col, CV_32FC3);
		ma7.create(row, col, CV_8UC3);
		ma8.create(row, col, CV_32FC3);
        ma9.create(row, col, CV_8UC3);  
	}
};


struct mapping_setup{
	cuda::GpuMat undistorted;
	cuda::GpuMat gpuMatSrc;
    cuda::GpuMat gpuMat;
    cuda::GpuMat gpuYuv;
	int flag;
	int num;
	int** mapGx;
	int** mapGy;
	mapping_setup(){};
	mapping_setup(int n, int height, int width){
		if(n == 1){
			undistorted.create(600, 4084, CV_8UC3);
		}
		else{
			undistorted.create(951, 1681, CV_8UC3);
		}
		gpuMatSrc.create(1080,1920, CV_8UC3);
        gpuMat.create(1080, 1920, CV_8UC3);
        gpuYuv.create(height*3/2, width, CV_8UC1);
		flag = 1;
		num = n;
		mapGx = NULL;
		mapGy = NULL;
	}
};

//编码前的预备空间，包括颜色空间转换、下载数据到cpu
struct out_data{
    cuda::GpuMat r_t;
    cuda::GpuMat res_t;
    Mat res_yuv;
    out_data() {}
    out_data(int h, int w){
        r_t.create(h, w, CV_8UC3);
        res_t.create((h-1)*3/2, w, CV_8UC1);
        res_yuv.create((h-1)*3/2, w, CV_8UC1);
        //res_yuv.at<uchar>(0, 0) = 1;
    }
};
