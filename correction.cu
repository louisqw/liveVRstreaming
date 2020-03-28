//main.cu
#include "global.h"
#include "merge.h"

//---------------------CUDA头文件----------------
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <opencv2/cudev/util/vec_traits.hpp>
//---------------------CUDA头文件----------------

using namespace std;

 

//fisheye correction CUDA kernel function
__global__ void GpuMapping(PtrStepSz<uchar3> dst,PtrStepSz<uchar3> src,int ** mapGx, int ** mapGy){
	int i;  // 列id，即x坐标
	int j;  // 行id，即y坐标
	
	i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i>=dst.cols)return;
	j = threadIdx.y + blockIdx.y * blockDim.y;
	if(j>=dst.rows)return;	
	//查表映射			  
	int tx = mapGx[j][i];
	int ty = mapGy[j][i];
	if(tx<src.rows && ty<src.cols && tx>=0 && ty>=0) 
		dst(j,i) = src(tx,ty);
	// 同步所有线程	  
	__syncthreads();
}

//YUV2RGB color conversion
__device__ uchar3 YUV2RGB(uchar Y, uchar U, uchar V){
	float Yt = Y;
	float Ut = U;
	float Vt = V;
	//float rf = Yt + 1.4*(Vt - 128);
	//float gf = Yt - 0.34*(Ut - 128) - 0.71*(Vt - 128);
	//float bf = Yt + 1.77*(Ut - 128);
	float rf = 1.164* (Yt-16) + 1.596*(Vt - 128);
	rf = (rf>0 && rf<255)?rf:rf<=0? 0:255;
	float gf = 1.164* (Yt-16) - 0.813*(Ut - 128) - 0.391*(Vt - 128);
	gf = (gf>0 && gf<255)?gf:gf<=0? 0:255; 
	float bf = 1.164* (Yt-16) + 2.018*(Ut - 128);
	bf = (bf>0 && bf<255)?bf:bf<=0? 0:255;
	uchar3 res = cv::cudev::VecTraits< uchar3 >::make((uchar)bf, (uchar)gf, (uchar)rf);
	return res;
}

//YUV2RGB color conversion CUDA kernel function
__global__ void YUV420P2RGB(PtrStepSz<uchar3> dst,PtrStepSz<uchar> src){
	int i = threadIdx.x + blockIdx.x * blockDim.x; // 列id，即x坐标
	int j = threadIdx.y + blockIdx.y * blockDim.y;  // 行id，即y坐标 
	int k = 0;
	for(int m = 0; m < 2; m++){
		//判断是否越界
		if(j >=src.rows/3 || i >= src.cols/2)break;
		k = j%2; 
		for(int n = 0; n < 2; n++){		   
			dst(2*j+n,2*i+m) =  YUV2RGB(
					src(2*j+n,2*i+m),//Y通道
					src(j/2+1080,i+k*960),//U通道
					src(j/2+1350,i+k*960)//V通道
					);
		}
	}
	// 同步所有线程
	__syncthreads();
}

//RGB2YUV color conversion CUDA kernel function
__global__ void RGB2YUV420P(PtrStepSz<uchar> dst, PtrStepSz<uchar> B, PtrStepSz<uchar>G, PtrStepSz<uchar> R){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = j%2;
	if(j >= R.rows/2 || i >= R.cols/2)return;
	for(int m = 0; m < 2; m++){
		for(int n = 0; n < 2; n++){ 
			//float Y = 0.299*R(2*j+n,2*i+m)+0.587*G(2*j+n,2*i+m)+0.114*B(2*j+n,2*i+m);
			float Y = 0.257*R(2*j+n,2*i+m)+ 0.564*G(2*j+n,2*i+m) +0.098*B(2*j+n,2*i+m) +16;
			dst(2*j+n,2*i+m) =(uchar)(Y>=0 && Y<=255)?Y:Y<0? 0:255;
		}
	}	
	//float U = -0.1687*R(2*j,2*i)-0.3313*G(2*j,2*i)+0.5*B(2*j,2*i)+128;
	//float V = 0.5*R(2*j,2*i)-0.4187*G(2*j,2*i)-0.0813*B(2*j,2*i)+128;
	float U = -0.148*R(2*j,2*i) - 0.291*G(2*j,2*i) + 0.439*B(2*j,2*i) + 128;
	float V = 0.439*R(2*j,2*i) - 0.368*G(2*j,2*i) - 0.071*B(2*j,2*i) +128;
	dst(j/2+1360,i+k*2042) =(uchar)(U>=0 && U<=255)?U:U<0? 0:255;
	dst(j/2+1700,i+k*2042) =(uchar)(V>=0 && V<=255)?V:V<0? 0:255;
	__syncthreads();	
}


//load correction mapping table from file 
void mapping_init(int num, int*** mapGx, int*** mapGy) {

	if(getCudaEnabledDeviceCount()==0){
		cerr<<"此OpenCV编译的时候没有启用CUDA模块"<<endl;
	}
	
	int i,j,k;
	int tx;
	int ty;
	int *** mapData;
	int ** mapGx_host;//内存用于存储显存中一维数组的暂用数组，之后将值赋给mapGx，完成GPU中二维数组构建
	int ** mapGy_host;
	
	//allocate host(RAM) space
	mapData = (int ***)malloc(2*sizeof(int **));
	if (NULL == mapData) ;
	for (i = 0; i<2; i++) {
		mapData[i] = (int **)malloc(951*sizeof(int *));
		if (NULL == mapData[i]) ;
		for (j = 0; j<951; j++) {
			mapData[i][j] = (int *)malloc(1681*sizeof(int));
			if(NULL == mapData[i][j]) ;
		}
	}
	
	//openfile
	char fname[256];
	sprintf(fname, "mapping_table/data%d.txt",num);
	ifstream in(fname);
	
	if (! in.is_open())
	{ cout << "Error opening file"<<endl; }
	in>>tx;
	in>>ty;
	
	//allocate device(VRAM) space, and copy X coordinates' mapping table from host to device
	cudaMalloc((void**)(mapGx), 951*sizeof(int*));
	mapGx_host = (int **)malloc(951*sizeof(int *));
	for (j = 0;j<951;j++) {
		for (k = 0;k<1681;k++){
			in>>mapData[0][j][k];
		}
		int* mapGx1;//一维GPU数组的指针
		cudaMalloc((void**)(&mapGx1), 1681*sizeof(int));
		cudaMemcpy((void*)(mapGx1), (void*)(mapData[0][j]), 1681*sizeof(int), cudaMemcpyHostToDevice);//将内存中的映射表赋给一维GPU数组
		mapGx_host[j] = mapGx1;
	}
	cudaMemcpy((void*)(*mapGx), (void*)(mapGx_host), 951*sizeof(int*), cudaMemcpyHostToDevice);

	//allocate device(VRAM) space, and copy Y coordinates' mapping table from host to device
	cudaMalloc((void**)(mapGy), 951*sizeof(int*));
	mapGy_host = (int **)malloc(951*sizeof(int *));
	for (j = 0;j<951;j++) {
		for (k = 0;k<1681;k++){
			in>>mapData[1][j][k];
		}
		int* mapGy1;//一维GPU数组的指针
		cudaMalloc((void**)(&mapGy1), 1681*sizeof(int));
		cudaMemcpy((void*)(mapGy1), (void*)(mapData[1][j]), 1681*sizeof(int), cudaMemcpyHostToDevice);//将内存中的映射表赋给一维GPU数组
		mapGy_host[j] = mapGy1;
	}
	cudaMemcpy((void*)(*mapGy), (void*)(mapGy_host), 951*sizeof(int*), cudaMemcpyHostToDevice);
	//cout<<"y done"<<endl;
	in.close();
	cout<<"map"<<num<<" loading complete"<<endl;
}

//load correction mapping table from file(top camera version)
void tmapping_init(int num, int*** mapGx, int*** mapGy) {
	if(getCudaEnabledDeviceCount()==0){
		cerr<<"此OpenCV编译的时候没有启用CUDA模块"<<endl;
	}
	
	int i,j,k;//计数器
	int tx;
	int ty;
	int *** mapData;
	int ** mapGx_host;//内存用于存储显存中一维数组的暂用数组，之后将值赋给mapGx，完成GPU中二维数组构建
	int ** mapGy_host;
	
	mapData = (int ***)malloc(2*sizeof(int **));
	if (NULL == mapData) ;
	for (i = 0; i<2; i++) {
		mapData[i] = (int **)malloc(600*sizeof(int *));
		if (NULL == mapData[i]) ;
		for (j = 0; j<600; j++) {
			mapData[i][j] = (int *)malloc(4084*sizeof(int));
			if(NULL == mapData[i][j]) ;
		}
	}
	
	ifstream in("mapping_table/data1.txt");
	
	if (! in.is_open())
	{ cout << "Error opening file"<<endl; }
	//else cout<<"loading data from data"<<num<<endl;
	in>>tx;
	in>>ty;
	
	//cout<<"start x map"<<endl;
	cudaMalloc((void**)(mapGx), 600*sizeof(int*));
	mapGx_host = (int **)malloc(600*sizeof(int *));
	for (j = 0;j<600;j++) {
		for (k = 0;k<4084;k++){
			in>>mapData[0][j][k];
		}
		int* mapGx1;//一维GPU数组的指针
		cudaMalloc((void**)(&mapGx1), 4084*sizeof(int));
		cudaMemcpy((void*)(mapGx1), (void*)(mapData[0][j]), 4084*sizeof(int), cudaMemcpyHostToDevice);//将内存中的映射表赋给一维GPU数组
		mapGx_host[j] = mapGx1;
	}
	cudaMemcpy((void*)(*mapGx), (void*)(mapGx_host), 600*sizeof(int*), cudaMemcpyHostToDevice);
	//cout<<"x done"<<endl;
	
	
	//cout<<"start y map"<<endl;
	cudaMalloc((void**)(mapGy), 600*sizeof(int*));
	mapGy_host = (int **)malloc(600*sizeof(int *));
	for (j = 0;j<600;j++) {
		for (k = 0;k<4084;k++){
			in>>mapData[1][j][k];
		}
		int* mapGy1;//一维GPU数组的指针
		cudaMalloc((void**)(&mapGy1), 4084*sizeof(int));
		cudaMemcpy((void*)(mapGy1), (void*)(mapData[1][j]), 4084*sizeof(int), cudaMemcpyHostToDevice);//将内存中的映射表赋给一维GPU数组
		mapGy_host[j] = mapGy1;
	}
	cudaMemcpy((void*)(*mapGy), (void*)(mapGy_host), 600*sizeof(int*), cudaMemcpyHostToDevice);
	//cout<<"y done"<<endl;
	in.close();
	cout<<"map"<<num<<" loading complete"<<endl;
}

//fisheye correction host function
void mapping(GpuMat& gpuMat,GpuMat& gpuMatSrc,int** mapGx,int** mapGy){
	dim3 threadsPerBlock(16, //一个block有多少列
				16); //一个block有多少行
	// 计算竖直需要多少个block
	uint block_num_vertical = gpuMat.rows/threadsPerBlock.x+1;
	// 计算水平需要多少个block
	uint block_num_horizontal = gpuMat.cols/threadsPerBlock.y+1;
	dim3 numBlocks(block_num_horizontal, // 列的方向的block数目
				   	block_num_vertical);  // 行的方向的block数目
	//call GPU mapping kernel function
	GpuMapping<<<numBlocks,threadsPerBlock>>>(gpuMat,gpuMatSrc,mapGx,mapGy);
	//synchronize all threads
	cudaDeviceSynchronize();
}

//YUV2RGB conversion host function
void YUV123(GpuMat& gpuMat,GpuMat& gpuMatSrc){
	dim3 threadsPerBlock(16, //一个block有多少列
				16); //一个block有多少行
	// 计算竖直需要多少个block
	uint block_num_vertical = gpuMat.rows/threadsPerBlock.y/2+1;
	//cout<<"block_num_vertical"<<block_num_vertical<<endl;
	// 计算水平需要多少个block
	uint block_num_horizontal = gpuMat.cols/threadsPerBlock.x/2+1;
	//cout<<"block_num_horizontal"<<block_num_horizontal <<endl;
	dim3 numBlocks(block_num_horizontal, // 列的方向的block数目
				   	block_num_vertical);  // 行的方向的block数目
	//call GPU mapping kernel function
	YUV420P2RGB<<<numBlocks,threadsPerBlock>>>(gpuMat,gpuMatSrc);
	//cudaDeviceSynchronize();
}

//RGB2YUV conversion host function
void YUV321(GpuMat& gpuMat,GpuMat& gpuMatSrc){
	//cout<<gpuMat.size()<<gpuMatSrc.size()<<endl;
	dim3 threadsPerBlock(16, //一个block有多少列
				16); //一个block有多少行
	// 计算竖直需要多少个block
	uint block_num_vertical = gpuMatSrc.rows/threadsPerBlock.y/2+1;
	//cout<<"block_num_vertical"<<block_num_vertical<<endl;
	// 计算水平需要多少个block
	uint block_num_horizontal = gpuMatSrc.cols/threadsPerBlock.x/2+1;
		//cout<<"block_num_horizontal"<<block_num_horizontal <<endl;
	dim3 numBlocks(block_num_horizontal, // 列的方向的block数目
				   	block_num_vertical);  // 行的方向的block数目
	GpuMat gpuMatYUV[3];
	split(gpuMatSrc,gpuMatYUV);
	//call GPU mapping kernel function
	RGB2YUV420P<<<numBlocks,threadsPerBlock>>>(gpuMat,gpuMatYUV[0],gpuMatYUV[1],gpuMatYUV[2]);
	//cudaDeviceSynchronize();
}



