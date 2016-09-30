#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>
#include "cublas_v2.h"
#include "cokus.cpp"
#include "cuda_util.h"
#include <cuda_runtime.h>
using namespace std;

const int NEIGHBOR = 8;//定义邻居个数
//const int DATA_BATCH = 512;//每次处理512个像素对应的数据

//CUDA初始化
bool InitCUDA(){
	int count;
	cudaGetDeviceCount(&count);
	if(count==0){
		fprintf(stderr,"There is no device.\n");
		return false;
	}
	int i;
	for (i =0; i<count;i++){
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){
			if(prop.major>=1){                                                                                                                                      break;
			}
		}
	}
	if(i==count){
		fprintf(stderr,"There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}


//copy数据到shared memory
__device__ void copy_data_to_shared(double * data, double * data_tmp, int length){
	for(int i=0; i<length; i++){
		data_tmp[i] = data[i];
	}

	__syncthreads();
}

//显卡处理数据
__global__ static void processing(int iter, double * data, int * train_index, double * processed_data, int x, int y, int z, int train_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	int idx = id * (NEIGHBOR+1) * z;//记录processed_data的开始位置
	if (id < train_size){
		for (int i=0; i<z; i++){
			for (int j=0; j<(NEIGHBOR+1); j++){
				processed_data[idx] = data[train_index[j + id*(NEIGHBOR+1)] + i * x*y];
				idx = idx + 1;	
			}
		}
	}
}

//数据预处理
int preprocess(double * data, double * labels, int x, int y, int z){
	double * gpu_data;//显存上存储原始数据
	double * gpu_processed_train;//显存上存储处理之后的数据
	int * gpu_train_index;//训练数据的索引

	//计算有标签像素的个数
	int train_size = 0;
	int * train_index = new int [x*y * (NEIGHBOR + 1)];//9行，x*y列。每列保存一个像素及其邻居的索引位置
	fprintf(stdout,"Preprocessing started!\n");

	for (int i=0; i<x*y; i++){
		if (labels[i] != 0){
			train_index[(NEIGHBOR/2) + train_size * (NEIGHBOR+1)] = i;//当前像素索引
			train_index[(NEIGHBOR/2) + train_size * (NEIGHBOR+1) - 1] = i - 1;
			train_index[(NEIGHBOR/2) + train_size * (NEIGHBOR+1) + 1] = i + 1;
			for(int j0=0;j0<3;j0++){
				train_index[j0 + train_size * (NEIGHBOR+1)] = i - 1 - x + j0;
				train_index[j0+6 + train_size * (NEIGHBOR+1)] = i - 1 + x + j0;
			}

			if((i % x) == 0){
				for (int j=0; j<3; j++)
					train_index[j*3 + train_size*(NEIGHBOR+1)] = train_index[j*3+2 + train_size*(NEIGHBOR+1)];
			}
			if((i % x) == (x-1)){
				for(int j=0;j<3;j++)
					train_index[j*3+2 + train_size*(NEIGHBOR+1)] = train_index[j*3 + train_size*(NEIGHBOR+1)];
			}
			if((i/x) == 0){
				for(int j=0;j<3;j++)
					train_index[j + train_size*(NEIGHBOR+1)] = train_index[j+6 + train_size*(NEIGHBOR+1)];
			}
			if((i/x) == (y-1)){
				for(int j=0;j<3;j++)
					train_index[j+6  + train_size*(NEIGHBOR+1)] = train_index[j + train_size*(NEIGHBOR+1)];
			}

			train_size = train_size + 1;
		}
	}
	fprintf(stdout,"train_size:%d\n",train_size);
	fprintf(stdout,"train_index[0]:%d %d %d %d,%d %d %d %d\n",train_index[0],train_index[1],train_index[2],train_index[3],train_index[5],train_index[6],train_index[7],train_index[8]);
	//fprintf(stdout,"train_index[10248]:%d %d %d %d,%d %d %d %d\n",train_index[9*10248],train_index[1+9*10248],train_index[2+9*10248],train_index[3+9*10248],train_index[5+9*10248],train_index[6+9*10248],train_index[7+9*10248],train_index[8+9*10248]);
	
	//int * train_index = new int [train_size * (NEIGHBOR + 1)];//train_size列，9行。每行保存一个像素及其邻居的索引位置

	fprintf(stdout,"Index computing completed!\n");

	//分配显存，拷贝数据到显存上
	SAFE_CALL(cudaMalloc((void **) &gpu_data, sizeof(double) * x * y * z));
	SAFE_CALL(cudaMemcpy(gpu_data, data, sizeof(double)* x * y * z, cudaMemcpyHostToDevice));

	SAFE_CALL(cudaMalloc((void **) &gpu_train_index, sizeof(int) * train_size * (NEIGHBOR+1)));
	SAFE_CALL(cudaMemcpy(gpu_train_index, train_index, sizeof(int) * train_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));

	SAFE_CALL(cudaMalloc((void **) &gpu_processed_train, sizeof(double) * train_size * (NEIGHBOR+1) * z));//每一批数据的大小

	int gridsize = 64;
	int blocksize = 1024;
	int threadnum = gridsize * blocksize; 
	double * processed_train = new double [train_size * (NEIGHBOR+1) * z];
	//预处理
	for (int iter=0; iter<=train_size/threadnum; iter++){
		processing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_train_index, gpu_processed_train, x, y, z, train_size);
	}
	cudaDeviceSynchronize();
	SAFE_CALL(cudaMemcpy(processed_train, gpu_processed_train, sizeof(double) * train_size * (NEIGHBOR+1) * z, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	fprintf(stdout,"Processed train data:%lf %lf %lf %lf\n",processed_train[0],processed_train[1],processed_train[2],processed_train[3]);
	
	return 0;
}

//主函数
int main(int argc, char * argv[])
{
  	if(!InitCUDA()){
		return 0;
	}
	printf("CUDA initialized.\n");

	clock_t start,end;

	double *trainset,*trainlabels;
	if(argc!=2){
		fprintf(stderr, "4 input arguments required!");
	}
	MATFile * datamat = matOpen(argv[1], "r");
	mxArray * train = matGetVariable(datamat,"DataSet");
	mxArray * labels = matGetVariable(datamat,"labels");

	trainset = (double*)mxGetData(train);
	trainlabels = (double*)mxGetData(labels);

	fprintf(stdout,"Data reading completed!\n");
	fprintf(stdout,"trainlabels:%lf %lf %lf %lf\n",trainlabels[87],trainlabels[88],trainlabels[89],trainlabels[90]);
	const mwSize  * dim;
	dim = mxGetDimensions(train);//获取trainset每维的元素个数
	fprintf(stdout,"Dimension:%d %d %d\n",dim[0],dim[1],dim[2]);

	start = clock();
	int te = preprocess(trainset, trainlabels, dim[0], dim[1], dim[2]);
	end = clock();
	double usetime = double(end - start);
	fprintf(stdout, "Using time of preprocessing:%lfs\n",usetime/CLOCKS_PER_SEC);
	return 0;
}
