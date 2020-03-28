#include "merge.h"
#include "global.h"

using namespace std;

//cvGPUinit threadsInit(GPU_NUM);

pthread_mutex_t srclock[5];
extern pthread_mutex_t lock;

deque<SRC_IMG> srcqueue[5];
extern deque<uint8_t*> framequeue;
extern vector<uint8_t*> out;         //内存池
extern vector<Mat> yuvsrc;            //源数据
extern vector<cuda::GpuMat> yuvsrcb;  //gpu源数据

//图像金字塔各层掩码，用于金字塔同层融合
extern vector<cuda::GpuMat> A_mask, B_mask, C_mask, D_mask, E_mask, U_mask;
extern vector<int> sign;
extern vector<vector<vector<int> > > poses;

//构建金字塔时的存储空间，固定存储大小及位置
extern layer* AP, *BP, *CP, *DP, *EP, *UP;
//
//模块并行时，数据的复制存储空间，固定存储大小及位置
extern layer* AP_t, *BP_t, *CP_t, *DP_t, *EP_t, *UP_t;
//

//logic control-------
extern int gn, cflag, rp, last, Tcnt;
//--------------------


//数据复制，以实现模块并行
void copy(layer* A, layer* B){
	A->p0.copyTo(B->p0);
	A->p1.copyTo(B->p1);
	A->p2.copyTo(B->p2);
	A->p3.copyTo(B->p3);
	A->p4.copyTo(B->p4);
	A->p5.copyTo(B->p5);
}

/*
//获取源数据主程序，并将数据上传GPU
void* get_src(void* arg){
	int cnt = 1;
	struct timeval start_get;
	while(true){
		if(!gn || cnt>Tcnt+4){
			usleep(1000);
			continue;
		}
		gettimeofday(&start_get, NULL);
		long st = (long)start_get.tv_sec * 1000000 + (long)start_get.tv_usec;
		for(int i = 0; i < RTSP_SOURCE_NUM; ++i) {
			pthread_mutex_lock(&srclock[i]);
			if(srcqueue[i].size() == 0){
				pthread_mutex_unlock(&srclock[i]);
				usleep(1000);
				continue;
			}
			memcpy(yuvsrc[i].data, srcqueue[i].front().img, IN_WIDTH*IN_HEIGHT*3/2);
			srcqueue[i].pop_front();
			pthread_mutex_unlock(&srclock[i]);
		}
		for(int i=0; i<RTSP_SOURCE_NUM; ++i)yuvsrcb[i].upload(yuvsrc[i]);
		//cout<<"-------------"<<endl;
		gettimeofday(&start_get, NULL);
		long en = (long)start_get.tv_sec * 1000000 + (long)start_get.tv_usec;
		cout<< cnt++ <<": extract time"<<en-st<<endl;
		gn=0;
	}
}
*/

//图像校正控制主程序
void* synCtrl(void* arg){
	GPUinit(GPU_NUM);
	struct timeval mmap;
	long start_map = 0;
	long end_map = 0;
	int cnt = 1;
	mapping_setup** setup = (mapping_setup**) arg;
	while(1){
		if(cflag == 0) {//gn==1 || cflag == 0 || cnt>Tcnt+3 3.28
			usleep(1000);
			continue;
		}
		gettimeofday(&mmap, NULL);
		start_map = (long)mmap.tv_sec * 1000000+ (long)mmap.tv_usec;

		/* 3.28 */
		//get src img
		for(int i=0; i<5; i++){
			
			//yuvsrcb[i].copyTo(setup[i]->gpuYuv);
			setup[i]->flag = 0;
		}
		

   		gn=1;
			
		//同步图像校正子线程
		while(1){
			if(setup[0]->flag&&setup[1]->flag&&setup[2]->flag&&setup[3]->flag&&setup[4]->flag)break;
			else usleep(1000);
		}

		gettimeofday(&mmap, NULL);
		end_map = (long)mmap.tv_sec * 1000000+ (long)mmap.tv_usec;
		cout<<cnt++<<":correction time:"<< end_map-start_map<<endl;			
		//cout<<"one group finished"<<endl;		
	cflag=0;
	}
}

//图像校正子线程，包含颜色空间转换YUV->RGB、图像校正
void* correction(void* mArg){
	GPUinit(GPU_NUM);
	mapping_setup* data = (mapping_setup*) mArg;
	//setCpu(data->num+6);
	while(1){
		if(data->flag) {usleep(1000);continue;}
		//color space conversion

		//YUV123(data->gpuMatSrc, data->gpuYuv);
		/* 3.28 */
		char *filename = new char[100]();
		sprintf(filename, "playground/%d.png", data->num);
		Mat tmp = imread(filename);
		data->gpuMatSrc.upload(tmp);

		//fisheye correction
		if (data->num == 1) mapping(data->undistorted,data->gpuMatSrc,data->mapGx,data->mapGy);
		else mapping(data->undistorted,data->gpuMatSrc,data->mapGx,data->mapGy);		
		data->flag = 1;
	}
}

//构建拉普拉斯金字塔
void* Pyramid(void* args) {  
	GPUinit(GPU_NUM);      
	//setCpu(*i);
	while(true){
		layer* lay=(layer*)((void**)args)[0];
		if(lay->tag==0){
			layer_support* lay_s1=(layer_support*)((void**)args)[1];
			layer_support* lay_s2=(layer_support*)((void**)args)[2];
			lay->img_src.convertTo(lay_s1->s0, CV_32F);
		
			cuda::pyrDown(lay_s1->s0, lay_s1->s1); 
			cuda::pyrDown(lay_s1->s1, lay_s1->s2); 
			cuda::pyrDown(lay_s1->s2, lay_s1->s3); 
			cuda::pyrDown(lay_s1->s3, lay_s1->s4); 
			cuda::pyrDown(lay_s1->s4, lay_s1->s5); 

			cuda::resize(lay_s1->s1, lay_s2->s0, lay_s1->s0.size());
			cuda::subtract(lay_s1->s0, lay_s2->s0, lay->p0);
			cuda::resize(lay_s1->s2, lay_s2->s1, lay_s1->s1.size());
			cuda::subtract(lay_s1->s1, lay_s2->s1, lay->p1);
			cuda::resize(lay_s1->s3, lay_s2->s2, lay_s1->s2.size());
			cuda::subtract(lay_s1->s2, lay_s2->s2, lay->p2);
			cuda::resize(lay_s1->s4, lay_s2->s3, lay_s1->s3.size());
			cuda::subtract(lay_s1->s3, lay_s2->s3, lay->p3);
			cuda::resize(lay_s1->s5, lay_s2->s4, lay_s1->s4.size());
			cuda::subtract(lay_s1->s4, lay_s2->s4, lay->p4);
			lay->p5=lay_s1->s5;
			lay->tag=1;
		}
		else{ usleep(1000); continue;}		
	}
}

//对多幅图像的金字塔，在同层间进行融合，获得全景帧的拉普拉斯金字塔
void* merge_part(void* args) { 
	GPUinit(GPU_NUM);
	while(true){
		int* sign=(int*)((void**)args)[8];
		if(*sign==0){
			cuda::GpuMat* imgA0=(cuda::GpuMat*)((void**)args)[0];
			cuda::GpuMat* imgB0=(cuda::GpuMat*)((void**)args)[1];
			cuda::GpuMat* imgC0=(cuda::GpuMat*)((void**)args)[2];
			cuda::GpuMat* imgD0=(cuda::GpuMat*)((void**)args)[3];
			cuda::GpuMat* imgE0=(cuda::GpuMat*)((void**)args)[4];
			cuda::GpuMat* imgU0=(cuda::GpuMat*)((void**)args)[5];
			cuda::GpuMat* r_s=(cuda::GpuMat*)((void**)args)[6];
			int* n=(int*)((void**)args)[7];
			merge_support* m_s=(merge_support*)((void**)args)[9];
			cuda::multiply(*imgA0, A_mask[*n], m_s->mc0);
			m_s->mc0.copyTo(m_s->ms0(Range(poses[0][0][*n], poses[0][1][*n]), Range(poses[0][2][*n], poses[0][3][*n])));
			cuda::multiply(*imgB0, B_mask[*n], m_s->mc1);
			m_s->mc1.copyTo(m_s->ms1(Range(poses[1][0][*n], poses[1][1][*n]), Range(poses[1][2][*n], poses[1][3][*n])));
			cuda::multiply(*imgC0, C_mask[*n], m_s->mc2);
			m_s->mc2.copyTo(m_s->ms2(Range(poses[2][0][*n], poses[2][1][*n]), Range(poses[2][2][*n], poses[2][3][*n])));
			cuda::multiply(*imgD0, D_mask[*n], m_s->mc3);
			m_s->mc3.copyTo(m_s->ms3(Range(poses[3][0][*n], poses[3][1][*n]), Range(poses[3][2][*n], poses[3][3][*n])));
			cuda::multiply(*imgE0, E_mask[*n], m_s->mc4);
			m_s->mc4.copyTo(m_s->ms4(Range(poses[4][0][*n], poses[4][1][*n]), Range(poses[4][2][*n], poses[4][3][*n])));
			cuda::multiply(*imgU0, U_mask[*n], m_s->mc5);
			m_s->mc5.copyTo(m_s->ms5(Range(poses[5][0][*n], poses[5][1][*n]), Range(poses[5][2][*n], poses[5][3][*n])));
			cuda::add(m_s->ms0, m_s->ms1, m_s->ms6);
			cuda::add(m_s->ms6, m_s->ms2, m_s->ms7);
			cuda::add(m_s->ms7, m_s->ms3, m_s->ms8);
			cuda::add(m_s->ms8, m_s->ms4, m_s->ms9);
			cuda::add(m_s->ms9, m_s->ms5, *r_s);
			*sign=1;
		}
		else {usleep(1000); continue;}  
	}
}

//基于全景帧的拉普拉斯金字塔，重构全景帧
void* merge(void* args) {
	GPUinit(GPU_NUM);
	cuda::GpuMat* img0=(cuda::GpuMat*)((void**)args)[0];
	cuda::GpuMat* img1=(cuda::GpuMat*)((void**)args)[1];
	cuda::GpuMat* img2=(cuda::GpuMat*)((void**)args)[2];
	cuda::GpuMat* img3=(cuda::GpuMat*)((void**)args)[3];
	cuda::GpuMat* img4=(cuda::GpuMat*)((void**)args)[4];
	cuda::GpuMat* img5=(cuda::GpuMat*)((void**)args)[5];
	merge_all_support* m_a = (merge_all_support*)((void**)args)[6];
	Mat* r=(Mat*)((void**)args)[7];

	cuda::resize(*img1, m_a->ma0, (*img0).size());
	cuda::add(*img0, m_a->ma0, m_a->ma1);
	cuda::resize(*img2, m_a->ma0, (*img0).size());
	cuda::add(m_a->ma1, m_a->ma0, m_a->ma2);
	cuda::resize(*img3, m_a->ma0, (*img0).size());
	cuda::add(m_a->ma2, m_a->ma0, m_a->ma3);
	cuda::resize(*img4, m_a->ma0, (*img0).size());
	cuda::add(m_a->ma3, m_a->ma0, m_a->ma4);
	cuda::resize(*img5, m_a->ma0, (*img0).size());
	cuda::add(m_a->ma4, m_a->ma0, m_a->ma5);

	m_a->ma7.convertTo(m_a->ma6, CV_32F);
	cuda::add(m_a->ma6, m_a->ma5, m_a->ma8);
	m_a->ma8.convertTo(m_a->ma9, CV_8U);

	/* 3.28 */
	if(access("result.png", F_OK)){
		imwrite("result.png", m_a->ma9);
	}

	return 0;
}

//金字塔构建的控制主程序，控制多幅图像的金字塔构建线程的同步
void* make_pm(void* args){
	GPUinit(GPU_NUM);
	mapping_setup** setup = (mapping_setup**) args;
	int cnt = 1;
	while(1){
	
		if(cflag == 1 || rp == 0){//  || cnt>Tcnt+2 3.28
			usleep(1000);
			continue;
		}
		
		struct timeval start;
		gettimeofday(&start, NULL);
		long start1 = (long)start.tv_sec * 1000000
			+ (long)start.tv_usec;
		setup[0]->undistorted(Range(0, 545), Range::all()).copyTo(UP->img_src(Range::all(), Range::all()));
		//
		setup[1]->undistorted(Range::all(), Range(840, 1608)).copyTo(AP->img_src(Range(90, 1041), Range::all()));
		setup[4]->undistorted(Range::all(), Range(139, 1483)).copyTo(BP->img_src(Range(90, 1041), Range::all()));
		setup[3]->undistorted(Range::all(), Range(142, 1486)).copyTo(CP->img_src(Range(90, 1041), Range::all()));
		setup[2]->undistorted(Range::all(), Range(145, 1489)).copyTo(DP->img_src(Range(90, 1041), Range::all()));
		setup[1]->undistorted(Range::all(), Range(72, 840)).copyTo(EP->img_src(Range(90, 1041), Range::all()));

		cflag = 1;
	
		AP->tag=0;
		BP->tag=0;
		CP->tag=0;
		DP->tag=0;
		EP->tag=0;
		UP->tag=0;
		
		//金字塔构建子线程同步
		while(1){                                    
			if(AP->tag==0 || BP->tag==0 || CP->tag==0 || DP->tag==0 || EP->tag==0 || UP->tag==0){
				usleep(1000);
				continue;
			}
			else break;
		}
		
		gettimeofday(&start, NULL);
		long end1 = (long)start.tv_sec * 1000000
			+ (long)start.tv_usec;
		cout<<cnt++<< ":pyramid:" << end1-start1 <<endl;
		
		rp=0;
	}
}

//融合控制主程序，控制多层融合线程同步，并控制全景帧重构线程
void* merge_main(void* args){
	GPUinit(GPU_NUM);
	int cnt  = 1;
	Mat* r=(Mat*)((void**)args)[7];

	while(1){
	
		if(rp == 1){//last==0 || rp == 1 || cnt > Tcnt+1 3.28
			usleep(1000);
			continue;
		}
	
		struct timeval start;
		gettimeofday(&start, NULL);
		long start1 = (long)start.tv_sec * 1000000
			+ (long)start.tv_usec;
		copy(AP, AP_t);
		copy(BP, BP_t);
		copy(CP, CP_t);
		copy(DP, DP_t);
		copy(EP, EP_t); 
		copy(UP, UP_t); 
		
        	rp=1;
		for(int i=0; i<6; ++i)
			sign[i]=0;

		 
		//金字塔多层融合子线程同步
		while(1){
			if(sign[0]==0 || sign[1]==0 || sign[2]==0 || sign[3]==0 || sign[4]==0 || sign[5]==0)
				usleep(1000);
			else break;
		}
		gettimeofday(&start, NULL);
		merge(args);              //全景帧重构线程
       
		long end1 = (long)start.tv_sec * 1000000
			+ (long)start.tv_usec;
		cout<<cnt++<< ":other  :" << end1-start1 <<endl;
        	last=0;
	}	 
}

/*

//编码前的准备，颜色空间转换RGB->YUV，数据下载到cpu
void* pre_encode(void* args) {
	merge_all_support* r  = (merge_all_support*)((void**)args)[0];
	out_data* res = (out_data*)((void**)args)[1];
	int n=0;
	int cnt = 1;
	struct timeval start_n; 
	gettimeofday(&start_n, NULL);
	long end2_n = (long)start_n.tv_sec * 1000000
			+ (long)start_n.tv_usec;
	while(1) {
		if(last==1) {
			usleep(1000);
			continue; 
		}
		
		gettimeofday(&start_n, NULL);
		long start1_n = (long)start_n.tv_sec * 1000000
			+ (long)start_n.tv_usec;
		
		r->ma9.copyTo(res->r_t);
		YUV321(res->res_t, res->r_t);
		res->res_t.download(res->res_yuv);
		memcpy(out[n], res->res_yuv.data, OUT_WIDTH*OUT_HEIGHT*3/2*sizeof(uint8_t)); 
		if(framequeue.size()>MAX_QUEUE_SIZE)
			continue;
		else{
			pthread_mutex_lock(&lock);
			framequeue.push_back(out[n]);
			n = (n+1)%20;
			pthread_mutex_unlock(&lock);
		}
		last = 1;
		Tcnt++;

	gettimeofday(&start_n, NULL);
	long end1_n = (long)start_n.tv_sec * 1000000
			+ (long)start_n.tv_usec;
	cout<<cnt<<":final	:"<<end1_n-start1_n<<endl;
	cout<<cnt++<<":one group total	:"<<end1_n-end2_n<<"\n"<<endl;
	end2_n = end1_n;	  
	} 
}
*/

