//#include<opencv2/opencv.hpp>
//#include<opencv2/gpu/gpu.hpp>
#include<sys/time.h>
#include<mat.h>
#include<pthread.h>
#include<string.h>
#include "global.h"
#include "merge.h"

#ifdef __cplusplus
extern "C"{
#endif
#include<libavcodec/avcodec.h>
#include<libavformat/avformat.h>
#include<math.h>

#ifdef __cplusplus
};
#endif

using namespace std;
using namespace cv;

cvGPUinit cvInit(GPU_NUM);


int bit_rate = BIT_RATE;
int frame_rate = FRAME_RATE;

deque<uint8_t*> framequeue;
vector<uint8_t*> out(20);         //内存池
vector<Mat> yuvsrc(5);            //源数据
vector<cuda::GpuMat> yuvsrcb(5);  //gpu源数据
pthread_mutex_t lock;
pthread_cond_t cond;

//thread function-------------------
extern void* decode(void* arg);
extern void* get_src(void* arg);	//yuv2rgb
extern void* correction(void* arg);	//fisheye correction
extern void* synCtrl(void* arg);	//fisheye control
extern void* Pyramid(void* args);	//pyramid construction
extern void* make_pm(void* args);	//pyramid control
extern void* merge_part(void* args);	//merge
extern void* merge_main(void* args);	//merge control
extern void* pre_encode(void* args);	//rgb2yuv
extern void* encode(void* filename);
//----------------------------------

pthread_t rthread1, rthread2, dthread[5], lthread, ethread;    //receive, decode and encode
pthread_t crThread[5];//fisheye correction thread 
pthread_t crSynThread;//fisheye correction control thread
pthread_t thread0, thread1[6];//5
pthread_t thread2, thread3[6];

//图像金字塔各层掩码，用于金字塔同层融合
vector<cuda::GpuMat> A_mask(6);             
vector<cuda::GpuMat> B_mask(6);
vector<cuda::GpuMat> C_mask(6);
vector<cuda::GpuMat> D_mask(6);
vector<cuda::GpuMat> E_mask(6);
vector<cuda::GpuMat> U_mask(6);


vector<int> sign(6, 1);
vector<vector<vector<int> > > poses;

//构建金字塔时的存储空间，固定存储大小及位置
layer* AP=new layer(1041, 768, 1);               
layer* BP=new layer(1041, 1344, 1);
layer* CP=new layer(1041, 1344, 1);
layer* DP=new layer(1041, 1344, 1);
layer* EP=new layer(1041, 768, 1);
layer* UP=new layer(545, 4084, 1);
//
//模块并行时，数据的复制存储空间，固定存储大小及位置
layer* AP_t=new layer(1041, 768, 0);
layer* BP_t=new layer(1041, 1344, 0);
layer* CP_t=new layer(1041, 1344, 0);
layer* DP_t=new layer(1041, 1344, 0);
layer* EP_t=new layer(1041, 768, 0);
layer* UP_t=new layer(545, 4084, 0);
//

//logic control-------
int gn=1;
int cflag = 1;
int rp= 1;
int last=1;
int Tcnt = 1;
//--------------------


//初始化，读取图像融合时的掩码文件
void read(mxArray *p, cuda::GpuMat& mask) {
	//double *ptr_data = (double*)mxGetPr(p);
	float *ptr_data = (float*)mxGetPr(p);
	const size_t *dims = mxGetDimensions(p);
	Mat temp(dims[0], dims[1], CV_32FC3);
	size_t subs[3];
	for (int i = 0; i < dims[0]; ++i) {
		subs[0] = i;
		for (int j = 0; j < dims[1]; ++j) {
			subs[1] = j;
			for (int k = 0; k < 3; ++k) {
				subs[2] = k;
				int index = mxCalcSingleSubscript(p, 3, subs);
				temp.at<Vec3f>(subs[0], subs[1])[k] =ptr_data[index];
			}
		}
	}
	mask.upload(temp);
}



//固定线程运行的cpu核
void setCpu(int n){
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(n, &mask);
	if (pthread_setaffinity_np(pthread_self(), sizeof(mask),&mask) < 0) {
			perror("pthread_setaffinity_np");
		}
}

int main(int argc, char* argv[]) {
	int oc;
	while((oc = getopt(argc, argv, "r:b:")) != -1) {
		switch(oc) {
			case 'r':frame_rate = atoi(optarg);break;
			case 'b':bit_rate = atoi(optarg);break;
		}
   	}
	
	struct timeval mread;
	gettimeofday(&mread, NULL);
	long start_r = (long)mread.tv_sec * 1000000
		+ (long)mread.tv_usec;
	
	//initialize streaming
	av_register_all();
	avcodec_register_all();
	avformat_network_init();
	
	//初始化，加载图像掩码
	MATFile *pmatfile = matOpen("./PIP2.mat", "r");
	vector<const char*> sA = { "maskA0","maskA1","maskA2","maskA3","maskA4","maskA5" };
	for (int i = 0; i < A_mask.size(); ++i) {
		mxArray *pA = matGetVariable(pmatfile, sA[i]);
		read(pA, A_mask[i]);
	}
	vector<const char*> sB = { "maskB0","maskB1","maskB2","maskB3","maskB4","maskB5" };
	for (int i = 0; i < B_mask.size(); ++i) {
		mxArray *pB = matGetVariable(pmatfile, sB[i]);
		read(pB, B_mask[i]);
	}
	vector<const char*> sC = { "maskC0","maskC1","maskC2","maskC3","maskC4","maskC5" };
	for (int i = 0; i < C_mask.size(); ++i) {
		mxArray *pC = matGetVariable(pmatfile, sC[i]);
		read(pC, C_mask[i]);
	}
	vector<const char*> sD = { "maskD0","maskD1","maskD2","maskD3","maskD4","maskD5" };
	for (int i = 0; i < D_mask.size(); ++i) {
		mxArray *pD = matGetVariable(pmatfile, sD[i]);
		read(pD, D_mask[i]);
	}
	vector<const char*> sE = { "maskE0","maskE1","maskE2","maskE3","maskE4","maskE5" };
	for (int i = 0; i < E_mask.size(); ++i) {
		mxArray *pE = matGetVariable(pmatfile, sE[i]);
		read(pE, E_mask[i]);
	}
	vector<const char*> sU = { "maskU0","maskU1","maskU2","maskU3","maskU4","maskU5" };
	for (int i = 0; i < U_mask.size(); ++i) {
		mxArray *pU = matGetVariable(pmatfile, sU[i]);
		read(pU, U_mask[i]);
	}
	matClose(pmatfile);
	
	for(int i=0; i<5; ++i){
		yuvsrc[i].create(IN_HEIGHT*3/2, IN_WIDTH, CV_8UC1);
		yuvsrcb[i].create(IN_HEIGHT*3/2, IN_WIDTH, CV_8UC1);
	}
	
	
	//四周图像在全景帧金字塔中的位置坐标，由特征点匹配获得
	vector<int> G_xl{ 320, 160, 80, 40, 20, 10 };         //公共x坐标 上
	vector<int> G_xr{ 1361, 681, 341, 171, 86, 43 };      //公共x坐标 下
	vector<int> A_yl{ 0, 0, 0, 0, 0, 0 };                 
	vector<int> A_yr{ 768, 384, 192, 96, 48, 24 };
	vector<int> B_yl{ 320, 160, 80, 40, 20, 10 };
	vector<int> B_yr{ 1664, 832, 416, 208, 104, 52 };
	vector<int> C_yl{ 1344, 672, 336, 168, 84, 42 };
	vector<int> C_yr{ 2688, 1344, 672, 336, 168, 84 };
	vector<int> D_yl{ 2368, 1184, 592, 296, 148, 74 };
	vector<int> D_yr{ 3712, 1856, 928, 464, 232, 116 };
	vector<int> E_yl{ 3316, 1658, 829, 415, 208, 104 };
	vector<int> E_yr{ 4084, 2042, 1021, 511, 256, 128 };
   
	//上图位置坐标
	vector<int> U_xl{ 0, 0, 0, 0, 0, 0 };
	vector<int> U_xr{ 545, 273, 137, 69, 35, 18 };
	vector<int> U_yl{ 0, 0, 0, 0, 0, 0 };
	vector<int> U_yr{ 4084, 2042, 1021, 511, 256, 128 };
		
	vector<vector<int> > midA;
	midA.push_back(G_xl);
	midA.push_back(G_xr);
	midA.push_back(A_yl);
	midA.push_back(A_yr);
	poses.push_back(midA);
	vector<vector<int> > midB;
	midB.push_back(G_xl);
	midB.push_back(G_xr);
	midB.push_back(B_yl);
	midB.push_back(B_yr);
	poses.push_back(midB);
	vector<vector<int> > midC;
	midC.push_back(G_xl);
	midC.push_back(G_xr);
	midC.push_back(C_yl);
	midC.push_back(C_yr);
	poses.push_back(midC);
	vector<vector<int> > midD;
	midD.push_back(G_xl);
	midD.push_back(G_xr);
	midD.push_back(D_yl);
	midD.push_back(D_yr);
	poses.push_back(midD);
	vector<vector<int> > midE;
	midE.push_back(G_xl);
	midE.push_back(G_xr);
	midE.push_back(E_yl);
	midE.push_back(E_yr);
	poses.push_back(midE);
	vector<vector<int> > midU;
	midU.push_back(U_xl);
	midU.push_back(U_xr);
	midU.push_back(U_yl);
	midU.push_back(U_yr);
	poses.push_back(midU);
	
	//correction initialize----------
	mapping_setup* setup[5];
	for(int i=0; i<5;i++){
		//initialize space for correction
		setup[i] = new mapping_setup(i+1, IN_HEIGHT, IN_WIDTH);
		//initialize 5 mapping table
		if(i == 0)tmapping_init(1,&setup[0]->mapGx,&setup[0]->mapGy);
		else mapping_init(i+1,&setup[i]->mapGx,&setup[i]->mapGy);
	}
	//--------------------------------
	//金字塔构建时辅助存储空间，固定存储大小及位置
	layer_support* AP_s1=new layer_support(1041, 768);
	layer_support* AP_s2=new layer_support(1041, 768);
	layer_support* BP_s1=new layer_support(1041, 1344);
	layer_support* BP_s2=new layer_support(1041, 1344);
	layer_support* CP_s1=new layer_support(1041, 1344);
	layer_support* CP_s2=new layer_support(1041, 1344);
	layer_support* DP_s1=new layer_support(1041, 1344);
	layer_support* DP_s2=new layer_support(1041, 1344);
	layer_support* EP_s1=new layer_support(1041, 768);
	layer_support* EP_s2=new layer_support(1041, 768);
	layer_support* UP_s1=new layer_support(545, 4084);
	layer_support* UP_s2=new layer_support(545, 4084);
	//---------------------------------------------------
	//同层融合时的辅助存储空间，固定存储大小和位置
	merge_support* M_s1=new merge_support(1041, 768, 1041, 1344, 545, 4084, 1361, 4084);
	merge_support* M_s2=new merge_support(489, 384, 489, 672, 113, 2042, 681, 2042);
	merge_support* M_s3=new merge_support(245, 192, 245, 336, 57, 1021, 341, 1021);
	merge_support* M_s4=new merge_support(123, 96, 123, 168, 29, 511, 171, 511);
	merge_support* M_s5=new merge_support(62, 48, 62, 84, 15, 256, 86, 256);
	merge_support* M_s6=new merge_support(31, 24, 31, 42, 8, 128, 43, 128);
	//全景帧重构的辅助存储空间，固定存储大小和位置
	merge_all_support* M_a = new merge_all_support(1361, 4084);
	//-----------------------------------

	gettimeofday(&mread, NULL);
	long end_r = (long)mread.tv_sec * 1000000
		+ (long)mread.tv_usec;
	cout<<"Load time:"<<end_r-start_r<<endl;
	for(int i=0; i<out.size(); ++i) {
		out[i] = (uint8_t*)malloc(OUT_WIDTH*OUT_HEIGHT*3/2);
	}
	pthread_mutex_init(&lock, NULL);
	pthread_cond_init(&cond, NULL);
	
	/* 3.28
	//源数据读取模块
	struct argument arg[5];
	for(int i=0; i<5; ++i)arg[i].index = i;
	strcpy(arg[0].url, RTSP_URL1);
	strcpy(arg[1].url, RTSP_URL2);
	strcpy(arg[2].url, RTSP_URL3);
	strcpy(arg[3].url, RTSP_URL4);
	strcpy(arg[4].url, RTSP_URL5);
	pthread_create(&dthread[0], NULL, decode, (void*)&arg[0]);
	pthread_create(&dthread[1], NULL, decode, (void*)&arg[1]);
	pthread_create(&dthread[2], NULL, decode, (void*)&arg[2]);
	pthread_create(&dthread[3], NULL, decode, (void*)&arg[3]);
	pthread_create(&dthread[4], NULL, decode, (void*)&arg[4]);
	pthread_create(&rthread2, NULL, get_src, NULL);
	*/

	//图像校正模块
	pthread_create(&crThread[0],NULL,correction,(void*) setup[0]);
	pthread_create(&crThread[1],NULL,correction,(void*) setup[1]);
	pthread_create(&crThread[2],NULL,correction,(void*) setup[2]);
	pthread_create(&crThread[3],NULL,correction,(void*) setup[3]);
	pthread_create(&crThread[4],NULL,correction,(void*) setup[4]);
	pthread_create(&crSynThread,NULL,synCtrl,(void*) setup);

	//金字塔构建模块
	int p1=0,p2=1,p3=2,p4=3,p5=4,p6=5;
	void* arga[4]={AP, AP_s1, AP_s2, &p1};
	void* argb[4]={BP, BP_s1, BP_s2, &p2};
	void* argc[4]={CP, CP_s1, CP_s2, &p3};
	void* argd[4]={DP, DP_s1, DP_s2, &p4};
	void* arge[4]={EP, EP_s1, EP_s2, &p5};
	void* argu[4]={UP, UP_s1, UP_s2, &p6};

	pthread_create(&thread0, NULL, make_pm, setup);	
	pthread_create(&thread1[0], NULL, Pyramid, arga);
	pthread_create(&thread1[1], NULL, Pyramid, argb);
	pthread_create(&thread1[2], NULL, Pyramid, argc);
	pthread_create(&thread1[3], NULL, Pyramid, argd);
	pthread_create(&thread1[4], NULL, Pyramid, arge);
	pthread_create(&thread1[5], NULL, Pyramid, argu);

	//图像融合模块
	cuda::GpuMat r0, r1, r2, r3, r4, r5,r;
	vector<int> flag{0, 1, 2, 3, 4, 5};
	r0.create(1361, 4084, CV_32FC3);
	r1.create(681, 2042, CV_32FC3);
	r2.create(341, 1021, CV_32FC3);
	r3.create(171, 511, CV_32FC3);
	r4.create(86, 256, CV_32FC3);
	r5.create(43, 128, CV_32FC3);
	uint8_t* res_last = (uint8_t*)malloc(OUT_WIDTH*OUT_HEIGHT*3/2);
	out_data* out_m = new out_data(1361, 4084); 
	void* arg0[10]={&AP_t->p0, &BP_t->p0, &CP_t->p0, &DP_t->p0, &EP_t->p0, &UP_t->p0, &r0, &flag[0], &sign[0], M_s1};
	void* arg1[10]={&AP_t->p1, &BP_t->p1, &CP_t->p1, &DP_t->p1, &EP_t->p1, &UP_t->p1, &r1, &flag[1], &sign[1], M_s2};
	void* arg2[10]={&AP_t->p2, &BP_t->p2, &CP_t->p2, &DP_t->p2, &EP_t->p2, &UP_t->p2, &r2, &flag[2], &sign[2], M_s3};
	void* arg3[10]={&AP_t->p3, &BP_t->p3, &CP_t->p3, &DP_t->p3, &EP_t->p3, &UP_t->p3, &r3, &flag[3], &sign[3], M_s4};
	void* arg4[10]={&AP_t->p4, &BP_t->p4, &CP_t->p4, &DP_t->p4, &EP_t->p4, &UP_t->p4, &r4, &flag[4], &sign[4], M_s5};
	void* arg5[10]={&AP_t->p5, &BP_t->p5, &CP_t->p5, &DP_t->p5, &EP_t->p5, &UP_t->p5, &r5, &flag[5], &sign[5], M_s6};
	void* arg6[8]={&r0, &r1, &r2, &r3, &r4, &r5, M_a, &r}; 
	void* arg7[3]={M_a, out_m, res_last};	

	pthread_create(&thread2, NULL, merge_main, arg6);	
	pthread_create(&thread3[0], NULL, merge_part, arg0);
	pthread_create(&thread3[1], NULL, merge_part, arg1);
	pthread_create(&thread3[2], NULL, merge_part, arg2);
	pthread_create(&thread3[3], NULL, merge_part, arg3);
	pthread_create(&thread3[4], NULL, merge_part, arg4);
	pthread_create(&thread3[5], NULL, merge_part, arg5);

	/* 3.28
	//图像编码推流模块
	pthread_create(&lthread, NULL, pre_encode, arg7);
	char* rtmpurl = RTMP_URL;
	pthread_create(&ethread, NULL, encode, (void*)rtmpurl);
	*/
	pthread_join(rthread2, NULL);
	pthread_join(thread0, NULL);
	pthread_join(thread2, NULL);
	//pthread_join(lthread, NULL); 3.28
	return 0;
}


