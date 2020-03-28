#all : livestream
CC = g++ -std=c++11
CUDA = nvcc
INCLUDE = /usr/local/include -I/usr/include/x86_64-linux-gnu
LIB = /usr/local/lib -L/usr/lib/x86_64-linux-gnu
MYLIB = -L/usr/local/x264/lib/libx264.a -lpthread -ldl -lrt -lmat -lmx -lcudart -lz -lavcodec -lavdevice -lavformat -lavutil -lswresample -lswscale
CVLIB = `pkg-config opencv --libs --cflags opencv`
MATLAB = -I/usr/local/MATLAB/R2018a/extern/include -L/usr/local/MATLAB/R2018a/bin/glnxa64/ -Wl,-rpath,/usr/local/MATLAB/R2018a/bin/glnxa64
CUDALIB = -L/usr/local/cuda/lib64
CFLAGS = -g 
	
livestream : main_thread.o livestream.o correction.o merge.o
	$(CC) $(CFLAGS) main_thread.o livestream.o correction.o merge.o -o livestream -I$(INCLUDE) -L$(LIB) $(MYLIB) $(CVLIB) $(MATLAB) $(CUDALIB)
	
livestream.o : livestream.c global.h
	$(CC) -I$(INCLUDE) $(CFLAGS) -L$(LIB) $(CVLIB) $(CUDALIB) -lmat -lmx -c livestream.c

correction.o : correction.cu global.h
	$(CUDA) -G -g -c correction.cu

merge.o : merge.cpp merge.h
	$(CC) -I$(INCLUDE) $(CFLAGS) -L$(LIB) $(LINKLIB) $(CVLIB) $(MATLAB) $(CUDALIB) -c merge.cpp

main_thread.o : main_thread.cpp global.h
	$(CC) -I$(INCLUDE) $(CFLAGS) -L$(LIB) $(LINKLIB) $(CVLIB) $(MATLAB) $(CUDALIB) -c main_thread.cpp

clean:
	rm -rf main_thread.o livestream livestream.o correction.o merge.o
