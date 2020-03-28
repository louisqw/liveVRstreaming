#include <stdio.h>

#define __STDC_CONSTANT_MACROS

#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>

#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

#include <libavutil/avassert.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libavutil/time.h>
#include <libavutil/pixfmt.h>
#ifdef __cplusplus
};
#endif
#include <deque> 
#include <iostream>
#include "global.h"
#define SFM_REFRESH_EVENT  (SDL_USEREVENT + 1)
#define STREAM_DURATION   20.0
//#define STREAM_FRAME_RATE 22
#define STREAM_PIX_FMT    AV_PIX_FMT_YUV420P 
//#define BIT_RATE   10000000

#define SCALE_FLAGS SWS_BICUBIC

//
int encN;
//debug
extern int frame_rate;
extern int bit_rate;
using namespace std;
extern pthread_mutex_t      srclock[5];
extern deque<SRC_IMG>     srcqueue[5];
extern pthread_mutex_t lock;
extern deque<uint8_t*> framequeue;
typedef struct OutputStream {
	AVStream *st;
	int64_t next_pts;
	int samples_count;
	AVFrame *frame;
	AVFrame *tmp_frame;
	float t, tincr, tincr2;
	struct SwsContext *sws_ctx;
	struct SwrContext *swr_ctx;
} OutputStream;

static int write_frame(AVFormatContext *fmt_ctx, const AVRational *time_base, AVStream *st, AVPacket *pkt)
{
	av_packet_rescale_ts(pkt, *time_base, st->time_base);
	pkt->stream_index = st->index;

	//	cout << "frame pts:" << pkt->pts << " " << pkt->dts << endl;
	struct timeval struc_start;
	gettimeofday(&struc_start, NULL);
	struct tm *p;
	p=gmtime(&struc_start.tv_sec);
	char s[10];
	strftime(s, 10, "%H:%M:%S", p);
	//printf("frame pts:%ld, %ld.%ld\n", pkt->pts, struc_start.tv_sec, struc_start.tv_usec/1000 );
	return av_interleaved_write_frame(fmt_ctx, pkt);
}

static void add_stream(OutputStream *ost, AVFormatContext *oc,
		AVCodec **codec,
		enum AVCodecID codec_id)
{
	AVCodecContext *c;
	int i;

	*codec = avcodec_find_encoder(codec_id);
	if (!(*codec)) {
		fprintf(stderr, "Could not find encoder for '%s'\n",
				avcodec_get_name(codec_id));
		exit(1);
	}

	ost->st = avformat_new_stream(oc, *codec);
	if (!ost->st) {
		fprintf(stderr, "Could not allocate stream\n");
		exit(1);
	}
	ost->st->id = oc->nb_streams-1;
	c = ost->st->codec;

	switch ((*codec)->type) {
		case AVMEDIA_TYPE_AUDIO:
			c->sample_fmt  = (*codec)->sample_fmts ?
				(*codec)->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
			c->bit_rate    = 64000;
			c->sample_rate = 44100;
			c->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
			if ((*codec)->supported_samplerates) {
				c->sample_rate = (*codec)->supported_samplerates[0];
				for (i = 0; (*codec)->supported_samplerates[i]; i++) {
					if ((*codec)->supported_samplerates[i] == 44100)
						c->sample_rate = 44100;
				}
			}
			c->channels        = av_get_channel_layout_nb_channels(c->channel_layout);
			c->channel_layout = AV_CH_LAYOUT_STEREO;
			if ((*codec)->channel_layouts) {
				c->channel_layout = (*codec)->channel_layouts[0];
				for (i = 0; (*codec)->channel_layouts[i]; i++) {
					if ((*codec)->channel_layouts[i] == AV_CH_LAYOUT_STEREO)
						c->channel_layout = AV_CH_LAYOUT_STEREO;
				}
			}
			c->channels        = av_get_channel_layout_nb_channels(c->channel_layout);
			ost->st->time_base = (AVRational){ 1, c->sample_rate };
			break;

		case AVMEDIA_TYPE_VIDEO:
			c->codec_id = codec_id;

			//c->flags |= CODEC_FLAG_QSCALE;
			//c->rc_min_rate = 0;
			//c->rc_max_rate = BIT_RATE;
			c->bit_rate = bit_rate;

			c->width = OUT_WIDTH;
			//c->height = OUT_HEIGHT;
			//c->height = OUT_HEIGHT-CUT_HEIGHT-TOP_CUT_HEIGHT;
			///c->height = OUT_HEIGHT-CUT_HEIGHT-TOP_CUT_HEIGHT+BOTTOM_HEIGHT;
			c->height = OUT_HEIGHT - CUT_HEIGHT - TOP_CUT_HEIGHT;
			ost->st->time_base = (AVRational){ 1, frame_rate };
			//ost->st->time_base = (AVRational){ 16, 347};//STREAM_FRAME_RATE };
			c->time_base       = ost->st->time_base;

			c->gop_size      = 12; 
			c->pix_fmt       = STREAM_PIX_FMT;
			if (c->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
				c->max_b_frames = 2;
			}
			if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
				c->mb_decision = 2;
			}
			av_opt_set(c->priv_data, "preset", "superfast", 0);  
			av_opt_set(c->priv_data, "tune", "zerolatency", 0);
			break;

		default:
			break;
	}

	if (oc->oformat->flags & AVFMT_GLOBALHEADER)
		c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
}
static AVFrame *alloc_picture(enum AVPixelFormat pix_fmt, int width, int height)
{
	AVFrame *picture;
	int ret;

	picture = av_frame_alloc();
	if (!picture)
		return NULL;

	picture->format = pix_fmt;
	picture->width  = width;
	picture->height = height;

	ret = av_frame_get_buffer(picture, 64);
	if (ret < 0) {
		fprintf(stderr, "Could not allocate frame data.\n");
		exit(1);
	}

	return picture;
}

static void open_video(AVFormatContext *oc, AVCodec *codec, OutputStream *ost, AVDictionary *opt_arg)
{
	int ret;
	AVCodecContext *c = ost->st->codec;
	AVDictionary *opt = NULL;

	av_dict_copy(&opt, opt_arg, 0);

	ret = avcodec_open2(c, codec, &opt);
	av_dict_free(&opt);
	if (ret < 0) {
		fprintf(stderr, "Could not open video codec\n");
		exit(1);
	}

	ost->frame = alloc_picture(c->pix_fmt, c->width, c->height);
	if (!ost->frame) {
		fprintf(stderr, "Could not allocate video frame\n");
		exit(1);
	}
	ost->tmp_frame = NULL;
	if (c->pix_fmt != AV_PIX_FMT_YUV420P) {
		ost->tmp_frame = alloc_picture(AV_PIX_FMT_YUV420P, c->width, c->height);
		if (!ost->tmp_frame) {
			fprintf(stderr, "Could not allocate temporary picture\n");
			exit(1);
		}
	}
}

static void IplImage_to_AVFrame(uint8_t* iplImage, AVFrame* avFrame, int frameWidth, int frameHeight, enum AVPixelFormat pix_fmt) { 
	struct SwsContext* img_convert_ctx = 0; int linesize[4] = {0, 0, 0, 0};

	img_convert_ctx = sws_getContext(frameWidth, frameHeight,
			AV_PIX_FMT_YUV420P,
			frameWidth,
			frameHeight,
			pix_fmt, SWS_BICUBIC, 0, 0, 0);
	if (img_convert_ctx != 0)
	{
		linesize[0] = 1 * frameWidth;
		sws_scale(img_convert_ctx, (const uint8_t * const *)&iplImage, linesize, 0, frameHeight, avFrame->data, avFrame->linesize);
		sws_freeContext(img_convert_ctx);
	}
}

//int delay = 1;
static int write_video_frame(AVFormatContext *oc, OutputStream *ost)
{
	struct timeval struc_start;
	//long sum = 0;
	//int64_t start_time = av_gettime();
	gettimeofday(&struc_start, NULL);
	long start = (long)struc_start.tv_sec*1000000
		+ (long)struc_start.tv_usec;
	int ret;
	AVCodecContext *c;
	AVFrame *frame;
	int got_packet = 0;

	c = ost->st->codec;
	frame = ost->frame;//get_video_frame(ost);
	ret = av_frame_make_writable(frame);
	if (ret < 0) {
		cout << "av_frame_make_writable error!" << endl;
		frame=NULL;
		return 0;
	}

	//cout << "encode size:" << framequeue.size() << endl;
	if(framequeue.size()==0) {
		usleep(1000);
		return 0;
	}
	pthread_mutex_lock(&lock);
	uint8_t* img = framequeue.front();
	framequeue.pop_front();
	pthread_mutex_unlock(&lock);
	frame->data[0] = img;
	///frame->data[1] = img+OUT_WIDTH*(OUT_HEIGHT-CUT_HEIGHT+BOTTOM_HEIGHT);
	///frame->data[2] = frame->data[1]+OUT_WIDTH*(OUT_HEIGHT-CUT_HEIGHT+BOTTOM_HEIGHT)/4;
	frame->data[1] = img + OUT_WIDTH * OUT_HEIGHT;
	frame->data[2] = frame->data[1]+OUT_WIDTH*OUT_HEIGHT/4;
	//frame->data[1] = img+OUT_WIDTH*(OUT_HEIGHT-CUT_HEIGHT-TOP_CUT_HEIGHT);
	//frame->data[2] = frame->data[1]+OUT_WIDTH*(OUT_HEIGHT-CUT_HEIGHT-TOP_CUT_HEIGHT)/4;
	frame->linesize[0] = OUT_WIDTH;
	frame->linesize[1] = OUT_WIDTH/2;
	frame->linesize[2] = OUT_WIDTH/2;
	//if(!frame) return 0;

	AVPacket pkt = { 0 };
	av_init_packet(&pkt);
	ret = avcodec_encode_video2(c, &pkt, frame, &got_packet);
	if (ret < 0) {
		fprintf(stderr, "Error encoding video frame\n");
		exit(1);
	}
	int64_t pts = pkt.pts+2;
	if (got_packet) {
		ret = write_frame(oc, &c->time_base, ost->st, &pkt);
		ost->frame->pts = ost->next_pts++;
		//free(frame->data);
	} else {
		ret = 0;
	}
	if(pts<0) pts=1;
	static int64_t start_time=av_gettime();  
	int64_t pts_time = (pts+2)*1000000/frame_rate;
	//delay++;
	int64_t now_time = av_gettime() - start_time;  
	//cout << pts  << "  " << pts_time/1000 << "  " << now_time/1000 << endl;
	//if (pts_time > now_time)  
	//	av_usleep(pts_time - now_time);  

	if (ret < 0) {
		fprintf(stderr, "Error while writing video frame\n");
		exit(1);
	}
	//printf("%x\n",frame->data[0]);
	//if(img!=NULL) {free(img);}
	gettimeofday(&struc_start, NULL);
	long end = (long)struc_start.tv_sec*1000000
		+ (long)struc_start.tv_usec;
    cout<<++encN<<"encode time:"<<end-start<<endl;
	return (frame || got_packet) ? 0 : 1;
}

static void close_stream(AVFormatContext *oc, OutputStream *ost)
{
	avcodec_close(ost->st->codec);
	av_frame_free(&ost->frame);
	av_frame_free(&ost->tmp_frame);
	sws_freeContext(ost->sws_ctx);
	swr_free(&ost->swr_ctx);
}

static AVFrame *alloc_audio_frame(enum AVSampleFormat sample_fmt,
		uint64_t channel_layout,
		int sample_rate, int nb_samples)
{
	AVFrame *frame = av_frame_alloc();
	int ret;

	if (!frame) {
		fprintf(stderr, "Error allocating an audio frame\n");
		exit(1);
	}

	frame->format = sample_fmt;
	frame->channel_layout = channel_layout;
	frame->sample_rate = sample_rate;
	frame->nb_samples = nb_samples;

	if (nb_samples) {
		ret = av_frame_get_buffer(frame, 0);
		if (ret < 0) {
			fprintf(stderr, "Error allocating an audio buffer\n");
			exit(1);
		}
	}

	return frame;
}

static void open_audio(AVFormatContext *oc, AVCodec *codec, OutputStream *ost, AVDictionary *opt_arg)
{
	AVCodecContext *c;
	int nb_samples;
	int ret;
	AVDictionary *opt = NULL;

	c = ost->st->codec;

	av_dict_copy(&opt, opt_arg, 0);
	ret = avcodec_open2(c, codec, &opt);
	av_dict_free(&opt);
	if (ret < 0) {
		fprintf(stderr, "Could not open audio codec\n");
		exit(1);
	}

	ost->t     = 0;
	ost->tincr = 2 * M_PI * 110.0 / c->sample_rate;
	ost->tincr2 = 2 * M_PI * 110.0 / c->sample_rate / c->sample_rate;

	if (c->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)
		nb_samples = 10000;
	else
		nb_samples = c->frame_size;

	ost->frame     = alloc_audio_frame(c->sample_fmt, c->channel_layout,
			c->sample_rate, nb_samples);
	ost->tmp_frame = alloc_audio_frame(AV_SAMPLE_FMT_S16, c->channel_layout,
			c->sample_rate, nb_samples);

	ost->swr_ctx = swr_alloc();
	if (!ost->swr_ctx) {
		fprintf(stderr, "Could not allocate resampler context\n");
		exit(1);
	}

	av_opt_set_int       (ost->swr_ctx, "in_channel_count",   c->channels,       0);
	av_opt_set_int       (ost->swr_ctx, "in_sample_rate",     c->sample_rate,    0);
	av_opt_set_sample_fmt(ost->swr_ctx, "in_sample_fmt",      AV_SAMPLE_FMT_S16, 0);
	av_opt_set_int       (ost->swr_ctx, "out_channel_count",  c->channels,       0);
	av_opt_set_int       (ost->swr_ctx, "out_sample_rate",    c->sample_rate,    0);
	av_opt_set_sample_fmt(ost->swr_ctx, "out_sample_fmt",     c->sample_fmt,     0);

	if ((ret = swr_init(ost->swr_ctx)) < 0) {
		fprintf(stderr, "Failed to initialize the resampling context\n");
		exit(1);
	}
}

static AVFrame *get_audio_frame(OutputStream *ost)
{
	AVFrame *frame = ost->tmp_frame;
	int j, i, v;
	int16_t *q = (int16_t*)frame->data[0];

	for (j = 0; j <frame->nb_samples; j++) {
		v = (int)(sin(ost->t) * 10000);
		for (i = 0; i < ost->st->codec->channels; i++)
			*q++ = v;
		ost->t     += ost->tincr;
		ost->tincr += ost->tincr2;
	}

	frame->pts = ost->next_pts;
	ost->next_pts  += frame->nb_samples;

	return frame;
}

static int write_audio_frame(AVFormatContext *oc, OutputStream *ost)
{		
	AVCodecContext *c;
	AVPacket pkt = { 0 }; // data and size must be 0;
	AVFrame *frame;
	int ret;
	int got_packet;
	int dst_nb_samples;

	av_init_packet(&pkt);
	c = ost->st->codec;

	frame = get_audio_frame(ost);

	ret = avcodec_encode_audio2(c, &pkt, frame, &got_packet);
	if (ret < 0) {
		fprintf(stderr, "Error encoding audio frame\n");
		exit(1);
	}

	if (got_packet) {
		ret = write_frame(oc, &c->time_base, ost->st, &pkt);
		if (ret < 0) {
			fprintf(stderr, "Error while writing audio frame\n");
			exit(1);
		}
	}

	return (frame || got_packet) ? 0 : 1;
}

void* encode(void* filename)
{
	OutputStream video_st = { 0 }, audio_st = { 0 };
	AVOutputFormat *fmt;
	AVFormatContext *oc;
	AVCodec *audio_codec, *video_codec;
	int ret;
	//
	encN = 0;
	//debug
	int have_video = 0, have_audio = 0;
	int encode_video = 0, encode_audio = 0;
	AVDictionary *opt = NULL;
    
    /**
    * ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo.
    */
    //av_dict_set(&opt, "preset", "fast", 0);  //av_opt_set(pCodecCtx->priv_data, "preset", "fast", 0)
    //av_dict_set(&opt, "tune", "zerolatency", 0);

	avformat_alloc_output_context2(&oc, NULL, "flv", (char*)filename);
	if (!oc)
		exit(1);

	fmt = oc->oformat;
	if (fmt->video_codec != AV_CODEC_ID_NONE) {
		//add_stream(&video_st, oc, &video_codec, fmt->video_codec);
		add_stream(&video_st, oc, &video_codec, AV_CODEC_ID_H264);
		have_video = 1;
		encode_video = 1;
	}
	//if (fmt->audio_codec != AV_CODEC_ID_NONE) {
	//        add_stream(&audio_st, oc, &audio_codec, AV_CODEC_ID_AAC);
	//        have_audio = 1;
	//        encode_audio = 1;
	//}

	video_st.st->codec->thread_count=40;
	if (have_video)
		open_video(oc, video_codec, &video_st, opt);

	//if (have_audio)
	//    open_audio(oc, audio_codec, &audio_st, opt);

	av_dump_format(oc, 0, (char*)filename, 1);

	if (!(fmt->flags & AVFMT_NOFILE)) {
		ret = avio_open(&oc->pb, (char*)filename, AVIO_FLAG_WRITE);
		if (ret < 0) {
			fprintf(stderr, "Could not open '%s'\n", filename);
			exit(1);
		}
	}

	ret = avformat_write_header(oc, &opt);
	if (ret < 0) {
		fprintf(stderr, "Error occurred when opening output file\n");
		exit(1);
	}

	//long id=0;
	struct timeval struc_start;
	//long sum = 0;
	int64_t start_time = av_gettime();
	gettimeofday(&struc_start, NULL);
	long start = (long)struc_start.tv_sec*1000000
		+ (long)struc_start.tv_usec;
	while (true) {
		encode_video = !write_video_frame(oc, &video_st);
	}
        //gettimeofday(&struc_start, NULL);
        //long end=(long)struc_start.tv_sec*1000000+(long)struc_start.tv_usec;
        //cout<<"encode time"<<end-start<<endl;
	cout << "encode end!" << endl;
	av_write_trailer(oc);

	if (have_video)
		close_stream(oc, &video_st);
	if (have_audio)
		close_stream(oc, &audio_st);

	if (!(fmt->flags & AVFMT_NOFILE))
		avio_closep(&oc->pb);

	avformat_free_context(oc);
}

void* decode(void* arg)
{
	AVFormatContext	*pFormatCtx;
	int				i, videoindex;
	AVCodecContext	*pCodecCtx;
	AVCodec			*pCodec;
	AVFrame	*pFrame;
	AVPacket *packet;
	int ret, got_picture;

	struct argument* arg1 = (struct argument*)arg;
	pFormatCtx = avformat_alloc_context();
	if(avformat_open_input(&pFormatCtx,(char*)arg1->url,NULL,NULL)!=0)
	{
		printf("Couldn't open input stream.\n");
		exit(1);
	}
	if(avformat_find_stream_info(pFormatCtx,NULL)<0)
	{
		printf("Couldn't find stream information.\n");
		exit(1);
	}
	videoindex=-1;
	for(i=0; i<pFormatCtx->nb_streams; i++) {
		if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO)
		{
			videoindex=i;
			break;
		}
	}

	if(videoindex==-1)
	{
		printf("Didn't find a video stream.\n");
		exit(1);
	}
	pCodecCtx=pFormatCtx->streams[videoindex]->codec;
	pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
	if(pCodec==NULL){
		printf("Codec not found.\n");
		exit(1);
	}
	pCodecCtx->thread_count = 2;
	if(avcodec_open2(pCodecCtx, pCodec,NULL)<0){
		printf("Could not open codec.\n");
		exit(1);
	}
	pFrame=av_frame_alloc();


	packet=(AVPacket *)av_malloc(sizeof(AVPacket));
	int width = pCodecCtx->width;
	int height = pCodecCtx->height;
	printf("width:%d, height:%d\n", width, height);
	AVRational frame_rate = pFormatCtx->streams[videoindex]->r_frame_rate;
	AVPicture pc;
	AVPicture* pic = &pc;
	avpicture_alloc(pic, AV_PIX_FMT_RGB24, width, height);
	AVFrame* frame = av_frame_alloc();
	struct SwsContext* sctx = sws_getContext(width, height, pCodecCtx->pix_fmt, width, height, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
	long id=0;
	struct timeval struc_start;
	long sum = 0;
	int linesize[4] = { IN_WIDTH*3, 0, 0, 0 };
	long pts = 1;
	int64_t start_time=av_gettime();  

	int delay[5] = {50,50,50,50,50};
	uint8_t* img = (uint8_t*)malloc(width*height*3);
	SRC_IMG srcimg;
	//int delay[5] = {110,122,100,108,100};
	for (;;) {
		if(av_read_frame(pFormatCtx, packet)>=0){
			if(packet->stream_index==videoindex){
				ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, packet);
				//cout << "frame pts:" << packet->pts << " " << arg1->index << endl;
				if(ret < 0){
					printf("Decode Error.\n");
					exit(1);
				}
				if(got_picture){
					if(srcqueue[arg1->index].size()>MAX_QUEUE_SIZE) {
					      av_free_packet(packet);
					      usleep(10000);
					      continue;
					}
					//uint8_t* img = (uint8_t*)malloc(width*height*3);//cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
					//sws_scale(sctx, pFrame->data, pFrame->linesize, 0, height,
					//		(uint8_t **) & img, linesize);
					//cout << pFrame->linesize[0] << " sss  " << pFrame->linesize[1] << "  " << pFrame->linesize[2] << endl;
					memcpy(img, pFrame->data[0], width*height);
					//if(arg1->index==4) {
					//	for(uint8_t* ti=img;ti<img+width*height;ti++) {
					//		
					//		printf("%d\t", *ti);
					//		*ti += 40;
					//	}
					//}
					memcpy(img+width*height, pFrame->data[1], width*height/4);
					memcpy(img+width*height*5/4, pFrame->data[2], width*height/4);
					gettimeofday(&struc_start, NULL);
					//SRC_IMG srcimg;
					srcimg.img = img;
					srcimg.time = struc_start.tv_sec*1000+struc_start.tv_usec/1000;
					if(delay[arg1->index]>0) 
						delay[arg1->index]--;
					else {
						pthread_mutex_lock(&srclock[arg1->index]);
						srcqueue[arg1->index].push_back(srcimg);
						pthread_mutex_unlock(&srclock[arg1->index]);
					}
					id++;
				}
			}
			av_free_packet(packet);
		}else{
			printf("Read Frame Error.\n");
			break;
		}
		int64_t pts_time = pts*40000;
		int64_t now_time = av_gettime() - start_time;  
		if (pts_time > now_time)  
			av_usleep(pts_time - now_time);  
		pts++;
	}

	av_frame_free(&pFrame);
	avcodec_close(pCodecCtx);
	avformat_close_input(&pFormatCtx);
	printf("Decode Exit.\n");
}
