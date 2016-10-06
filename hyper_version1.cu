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

const int KER_NUM = 20;//卷积核数量
const int P_NUM = 3;//每次卷积的层数
const int LEAP = 2;//跳数
const int GP_NUM = 5;//maxpooling每组的个数
const int NEU_NUM1 = 100;
const int NEU_NUM2 = 16;//输出层神经元个数
const int NEIGHBOR = 8;//定义邻居个数
const double LEARN_RATE = 0.01;
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
__device__ void copy_data_to_shared(double * data, double * data_tmp,int head, int length){
	for(int i=0; i<length; i++){
		data_tmp[i] = data[i+head];
	}

	__syncthreads();
}

//GPU端负责卷积
__global__ static void convol(int iter,int i0,double * train,double * kernel,double * re,int x,int y,int z,int re_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;//保存当前线程编号

	//每个线程负责一个卷积核与一个3*3*hight柱状图像的卷积
	if (id < KER_NUM){
		extern __shared__ double train_tmp[];
		//__shared__ double train_tmp[9*200];
		int st = i0 * x * y * z;

		copy_data_to_shared(train,train_tmp,st,x*y*z);//复制train到shared memory中

		/*double * ker = new double [x*y*P_NUM];//载入对应的kernel到寄存器
		for(int i=0; i<x*y*P_NUM; i++){
			ker[i] = kernel[id*x*y*P_NUM + i];
		}*/

		int dim_x = 0, dim_y = 0, dim_z = 0;//初始位置为(0,0,0)

		double mid;
		int i_1=0;
		for(; dim_z+P_NUM-1 < z; dim_z=dim_z+LEAP){//每次跳LEAP层
			mid = 0.0;

			for(int i_0=0;i_0<P_NUM;i_0++){//每次进行3*3*P_NUM的像素块的卷积
				for(int i=0;i<x;i++){
					for(int j=0;j<y;j++){
						mid =mid + train_tmp[dim_x+j + (dim_y+i) * x + (dim_z+i_0)*x*y] * kernel[j + i*x + i_0*x*y + id*x*y*P_NUM];
					}
				}
			}

			re[i_1 + id * re_size] =2 * (1/(1 + 1/exp(2*mid))) - 1;//激活函数tanh
			i_1 ++;
		}

	}
}

//GPU端进行下采样
__global__ static void maxpooling(int iter,double * re,double * mre,int * mre_index,int re_size,int mre_num){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
       	int id = tid + iter * threadNum; 
	
	//int res = re_size, mres = mre_num;
	//extern __shared__ double re_tmp[];
	//copy_data_to_shared(re, re_tmp, 0, re_size*KER_NUM);

	if(id < KER_NUM){
		double mid;
		int mid_index;
		for(int i=0; i<mre_num; i++){
			mid = re[i*GP_NUM + id*re_size];//存放每组第一个值
			mid_index = i*GP_NUM + id*re_size;
			for(int j=i*GP_NUM+1; j<(i+1)*GP_NUM && j<re_size; j++){
				if(mid < re[j + id*re_size]){
					mid = re[j + id*re_size];
					mid_index = j+id*re_size;
				}
			}
			mre[i + id * mre_num] = mid;
			mre_index[i + id * mre_num] = mid_index;
		}
	}
}

//全连接层,每个线程负责一个神经元输出结果的计算
__global__ static void fullconnect(int iter,double * mre,double * omega,double * bias,double * F1,int mre_size){
	int tid = blockIdx.x * blockDim.x +threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	if(id < NEU_NUM1){
		//复制mre数组到共享内存
		//__shared__ double mre_tmp[50 * KER_NUM];
	        extern __shared__ double mre_tmp[];	
		copy_data_to_shared(mre,mre_tmp,0,mre_size);
		
		//计算神经元的输出
		double mid=0;
		int j; 
		for(int i=id*mre_size; i<(id+1)*mre_size; i++){
			j = 0;
			mid = mid + omega[i] * mre_tmp[j];
			j = j + 1;
		}
		F1[id] = 1/(1 + 1/exp(mid + bias[id]));//激活函数sigmod
	}
}

//输出层，每个线程负责一个神经元输出结果的计算
__global__ static void output(int iter, double * F1, double * omega2, double * bias, double * O2){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	if(id < NEU_NUM2){
		//复制F1到共享内存中
		__shared__ double F1_tmp[NEU_NUM1];
		copy_data_to_shared(F1, F1_tmp, 0, NEU_NUM1);

		//计算神经元的输出
		double mid = 0;
		int j;
		for(int i=id*NEU_NUM1; i<(id+1)*NEU_NUM1; i++){
			j = 0;
			mid = mid + omega2[i] * F1_tmp[j];
			j = j + 1;
		}
		O2[id] = 1/(1 + 1/exp(mid + bias[id]));//激活函数sigmod
	}
}

/*反向传播*/
//输出层
__global__ static void bp_output(int iter,int train_idx, double * labels, double * O2, double * bias2, double * delta_L_a, double * delta_L_z)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;
	if(id < NEU_NUM2){
		delta_L_a[id] = -(labels[id + train_idx * NEU_NUM2] - O2[id]);
		delta_L_z[id] = delta_L_a[id] * O2[id] *(1 - O2[id]);
		bias2[id] = bias2[id] - delta_L_z[id]*LEARN_RATE;
	}
}
//全连接层
__global__ static void bp_fullconnect(int iter, double * omega2,double * bias1, double * F1, double * delta_L_a, double * delta_L_z, double *delta_f_a, double * delta_f_z)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;
	if(id < NEU_NUM1){
		double mid = 0;
		double delta_f_w;

		for(int i=0; i<NEU_NUM2; i++){
			mid = mid + omega2[i + id*NEU_NUM2] * delta_L_z[i];
			//delta_f_b[i] = delta_L_z[i];
			delta_f_w = F1[id] * delta_L_z[i];

			omega2[i + id*NEU_NUM2] = omega2[i + id*NEU_NUM2] - LEARN_RATE * delta_f_w;
			//bias2[i] = bias2[i] - LEARN_RATE*delta_f_b[i];
		}
		delta_f_a[id] = mid;
		delta_f_z[id] = delta_f_a[id] * F1[id] * (1 - F1[id]);
		bias1[id] = bias1[id] - LEARN_RATE * delta_f_z[id];
	}
}
//maxpooling层（卷积层）
__global__ static void bp_maxpooling(int iter, int mre_size,int *mre_index, double * omega1,double *mre, double * delta_f_a, double * delta_f_z, double * delta_m_a, double * delta_2, double * delta_22)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int threadNum = blockDim.x * gridDim.x;
        int id = tid + iter * threadNum;
	if(id < mre_size){
		double mid = 0;
		double delta_m_w;
		for(int i=0; i<NEU_NUM1; i++){
			mid = mid + omega1[i + id*NEU_NUM1] * delta_f_z[i];
			//delta_2[i + id*NEU_NUM1] = mid;
			delta_m_w = mre[id] * delta_f_z[i];
			omega1[i + id*NEU_NUM1] = omega1[i + id*NEU_NUM1] - LEARN_RATE * delta_m_w;
		}
		delta_m_a[id] = mid;
		delta_2[id] = delta_m_a[id];
		
		//int idx = mre_index[id];
		delta_22[mre_index[id]] = delta_2[id];
	}
}
//输入层
/*__global__ static void bp_map_convol(int iter,int mre_num, int * max_idx,double * re, double * delta_2, double * delta_22){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;
	if(id < mre_num * KER_NUM){
		int idx = max_idx[id];
		for(int i=0; i<NEU_NUM1; i++){
			delta_22[i + idx*NEU_NUM1] = delta_2[i +id*NEU_NUM1];
			delta_22[i + idx*NEU_NUM1] = delta_22[i + idx*NEU_NUM1] * (1 - re[idx]) * (1 + re[idx]);
		}

		extern __shared__ double train_tmp[];
		copy_data_to_shared(train_tmp, data, 9*z*train_idx, 9*z);

		double delta_z = 0;
		double delta_w = 0;
		int ii=0;
		for(int i=0; i<re_size; i++){//处理一个卷积核的所有计算结果
			ii = i*LEAP*9;
			for(int j=0; j<3*3*P_NUM; j++){
				delta_z = delta_m_a[i/GP_NUM + id*mre_num] * (1 - train_tmp[ii+j] * train_tmp[ii+j]);
				delta_w = delta_z * train_tmp[ii+j];
				kernel[j + id*9*P_NUM] = kernel[j + id*9*P_NUM] - LEARN_RATE*delta_w;
			}
		}
	}
}*/
//计算并更新kernel
__global__ static void bp_update_kernel(int iter,int i0, int x, int y, int z, int mre_num,int re_size, int * mre_index, double * delta_22, double * data, double * delta_k_w,double * kernel)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum =blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;
	if(id < KER_NUM){
		extern __shared__ double train_tmp[];
		copy_data_to_shared(data, train_tmp, x*y*z*i0, x*y*z);

		for (int i=0; i<mre_num; i++){
			int idx = mre_index[i + id*mre_num];
			int n = idx % re_size;//对应卷积的第n块数据
			int head = x*y*LEAP*n;
			for(int j=0; j<x*y*P_NUM; j++){
				delta_k_w[j] = delta_k_w[j] + /*delta_22[idx] **/ train_tmp[j+head];
			}
		
		}

		for(int i=0;i<x*y*P_NUM;i++){
			delta_k_w[i] = delta_k_w[i]/mre_num;
			kernel[id*x*y*P_NUM+i] = kernel[id*x*y*P_NUM+i] - LEARN_RATE*delta_k_w[i];
		}
		//double delta_k_b = delta_22[id + idx*re_size*NEU_NUM1];
	}
}

//数据预处理
__global__ static void processing(int iter, double * data, int * train_index, double * processed_data, int x, int y, int z, int train_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNum = blockDim.x * gridDim.x;
	int id = tid + iter * threadNum;

	//int idx = id * (NEIGHBOR+1) * z;//记录processed_data的开始位置
	if (id < train_size){
		int idx = id * (NEIGHBOR+1) * z;
		for (int i=0; i<z; i++){
			for (int j=0; j<(NEIGHBOR+1); j++){
				processed_data[idx] = data[train_index[j + id*(NEIGHBOR+1)] + i * x*y];
				idx = idx + 1;	
			}
		}
	}
}

double lossfunction(double * output, double * labels, int idx){
	double l = 0;
	for(int i=0; i<NEU_NUM2; i++){
		l = l + (output[i] - labels[i + idx*NEU_NUM2]) * (output[i] - labels[i + idx*NEU_NUM2]);
	}
	l = l/2;
	return l;
}

//训练
int training(double * data, double * labels, int x, int y, int z){
	double * gpu_data;//显存上存储原始数据
	double * gpu_processed_train;//显存上存储处理之后的数据
	int * gpu_train_index;//训练数据的索引
	double * gpu_processed_labels;

	//计算有标签像素的个数
	int train_size = 0;
	int * train_index = new int [x*y * (NEIGHBOR + 1)];//9行，x*y列。每列保存一个像素及其邻居的索引位置
	double * processed_labels = new double [x*y * NEU_NUM2];
	for (int i=0; i<x*y; i++){
		if (labels[i] != 0){
			train_index[(NEIGHBOR/2) + train_size * (NEIGHBOR+1)] = i;//当前像素索引
			train_index[(NEIGHBOR/2) + train_size * (NEIGHBOR+1) - 1] = i - 1;
			train_index[(NEIGHBOR/2) + train_size * (NEIGHBOR+1) + 1] = i + 1;
			for(int j0=0;j0<3;j0++){
				train_index[j0 + train_size * (NEIGHBOR+1)] = i - 1 - x + j0;
				train_index[j0+6 + train_size * (NEIGHBOR+1)] = i - 1 + x + j0;
			}

			if((i % x) == 0){//第一行
				for (int j=0; j<3; j++)
					train_index[j*3 + train_size*(NEIGHBOR+1)] = train_index[j*3+2 + train_size*(NEIGHBOR+1)];
			}
			if((i % x) == (x-1)){//最后一行
				for(int j=0;j<3;j++)
		       			train_index[j*3+2 + train_size*(NEIGHBOR+1)] = train_index[j*3 + train_size*(NEIGHBOR+1)];
			}
			if((i/x) == 0){//第一列
				for(int j=0;j<3;j++)
					train_index[j + train_size*(NEIGHBOR+1)] = train_index[j+6 + train_size*(NEIGHBOR+1)];
			}
			if((i/x) == (y-1)){//最后一列
				for(int j=0;j<3;j++)
					train_index[j+6  + train_size*(NEIGHBOR+1)] = train_index[j + train_size*(NEIGHBOR+1)];
			}

			int mid = int(labels[i])-1 + train_size*NEU_NUM2;
			processed_labels[mid] = 1;
			train_size = train_size + 1;
		}
	}
	fprintf(stdout,"train_size:%d\n",train_size);
	fprintf(stdout,"train_index[0]:%d %d %d %d,%d %d %d %d\n",train_index[0],train_index[1],train_index[2],train_index[3],train_index[5],train_index[6],train_index[7],train_index[8]);
	//fprintf(stdout,"train_index[5211]:%d %d %d %d,%d %d %d %d\n",train_index[46890],train_index[46891],train_index[46892],train_index[46893],train_index[46895],train_index[46896],train_index[46897],train_index[46898]);
	
	//int * train_index = new int [train_size * (NEIGHBOR + 1)];//train_size列，9行。每行保存一个像素及其邻居的索引位置


	//分配显存，拷贝数据到显存上
	SAFE_CALL(cudaMalloc((void **) &gpu_data, sizeof(double) * x * y * z));
	SAFE_CALL(cudaMemcpy(gpu_data, data, sizeof(double)* x * y * z, cudaMemcpyHostToDevice));

	SAFE_CALL(cudaMalloc((void **) &gpu_train_index, sizeof(int) * train_size * (NEIGHBOR+1)));
	SAFE_CALL(cudaMemcpy(gpu_train_index, train_index, sizeof(int) * train_size * (NEIGHBOR+1), cudaMemcpyHostToDevice));

	SAFE_CALL(cudaMalloc((void **) &gpu_processed_train, sizeof(double) * train_size * (NEIGHBOR+1) * z));//每一批数据的大小

	int gridsize = 64;
	int blocksize = 1024;
	int threadNum = gridsize * blocksize; 
	//double * processed_train = new double [train_size * (NEIGHBOR+1) * z];
	//预处理
	for (int iter=0; iter<=train_size/threadNum; iter++){
		processing<<<gridsize,blocksize>>>(iter, gpu_data, gpu_train_index, gpu_processed_train, x, y, z, train_size);
	}
	cudaDeviceSynchronize();
	//SAFE_CALL(cudaMemcpy(processed_train, gpu_processed_train, sizeof(double) * train_size * (NEIGHBOR+1) * z, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaFree(gpu_data));
	SAFE_CALL(cudaFree(gpu_train_index));
	cudaDeviceSynchronize();
	//fprintf(stdout,"Processed train data:%lf %lf %lf %lf\n",processed_train[0],processed_train[1],processed_train[2],processed_train[3]);
	
	//前向传播
	double * kernel = new double [(NEIGHBOR+1)*P_NUM*KER_NUM];

	//随机生成kernekl数组
	for(int i=0; i<(NEIGHBOR+1)*P_NUM*KER_NUM; i++){
		kernel[i] = 2 * (rand()/(double)(RAND_MAX)) - 1 ;
		if(kernel[i] == 0 || kernel[i] == -1 || kernel[i] == 1)
			kernel[i] = 0.001;
	}
	
	//计算每次卷积的结果个数
	int re_size = 0;
	for (int i=0; i+P_NUM-1<z; i+=LEAP){
		re_size ++;
	}

	double * re = new double [re_size * KER_NUM];
	fprintf(stdout,"Size of re:%d\n",re_size);

	int mre_num = re_size/GP_NUM + 1;
	if(re_size/GP_NUM == 0){
		mre_num = re_size / GP_NUM;
	}
	fprintf(stdout,"mre_num:%d\n",mre_num);
	int mre_size = mre_num * KER_NUM;
	int ome_num1 = mre_num * KER_NUM * NEU_NUM1;//第一层网络的输入权重个数
	int ome_num2 = NEU_NUM1 * NEU_NUM2;//输出层的权重个数
	
	//double * gpu_labels;
	double * gpu_kernel;
	double * gpu_re;//存放卷积结果
	double * gpu_mre;//存放maxpooling结果
	int * gpu_mre_index;//存放每组最大值的索引
	double * gpu_omega1;//第一层网络的输入权重
	double * gpu_F1;//第一层神经元的输出
	double * gpu_bias1;
	double * gpu_omega2;
	double * gpu_O2;
	double * gpu_bias2;
	double * gpu_delta_La;
	double * gpu_delta_Lz;
	double * gpu_delta_fa;
	double * gpu_delta_fz;
	double * gpu_delta_ma;
	double * gpu_delta_mz;
	double * gpu_delta_2;
	double * gpu_delta_22;
	double * gpu_delta_kw;
	//double * gpu_delta_ia;
	//double * gpu_delta_iz;
	double * delta_22 = new double [re_size*KER_NUM];

	//复制标签
	SAFE_CALL(cudaMalloc((void**) &gpu_processed_labels, sizeof(double) * train_size * NEU_NUM2));
	SAFE_CALL(cudaMemcpy(gpu_processed_labels,processed_labels,sizeof(double) * train_size * NEU_NUM2,cudaMemcpyHostToDevice));
	//复制随机初始化的kernel数组
	SAFE_CALL(cudaMalloc((void**) &gpu_kernel,sizeof(double) * (NEIGHBOR+1) * P_NUM * KER_NUM));
	SAFE_CALL(cudaMemcpy(gpu_kernel,kernel,sizeof(double) * (NEIGHBOR+1) * P_NUM * KER_NUM,cudaMemcpyHostToDevice));
	//卷积结果存入gpu_re，分配显存
	SAFE_CALL(cudaMalloc((void **) &gpu_re,sizeof(double) * re_size * KER_NUM));
	//输出层偏导数
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_La, sizeof(double) * NEU_NUM2));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_Lz, sizeof(double) * NEU_NUM2));
	//全连接层偏导数
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_fa, sizeof(double) * NEU_NUM1));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_fz, sizeof(double) * NEU_NUM1));
	//maxpooling
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_ma, sizeof(double) * mre_size));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_mz, sizeof(double) * mre_size));
	//输入层
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_2, sizeof(double) * mre_size));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_22,sizeof(double) * re_size * KER_NUM));
	SAFE_CALL(cudaMemcpy(gpu_delta_22, delta_22, sizeof(double) * re_size * KER_NUM, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMalloc((void **) &gpu_delta_kw, sizeof(double) * (NEIGHBOR+1) *P_NUM));

	double * omega1 = new double [ome_num1];
	double * omega2 = new double [ome_num2];
	double * bias1 = new double [NEU_NUM1];
	double * bias2 = new double [NEU_NUM2];

	//随机生成Omega1
	for(int i=0; i<ome_num1; i++){
		omega1[i] = 2 * (rand()/(double)(RAND_MAX)) - 1 ;
	        if(omega1[i] == 0 || omega1[i] == -1 || omega1[i] == 1)
			omega1[i] = 0.001;
	}
	//随机生成bias1
	for(int i=0; i<NEU_NUM1; i++){
		bias1[i] = 2*(rand()/(double)(RAND_MAX)) - 1;
	}

	//随机生成Omega2
	for(int i=0; i<ome_num2; i++){
		omega2[i] = 2 * (rand()/(double)(RAND_MAX)) - 1;
		if(omega2[i] ==0 || omega2[i] == 1 || omega2[i] ==-1)
			omega2[i] = 0.001;
	}
	fprintf(stdout, "Bias1: %lf %lf %lf\n",bias1[0],bias1[1],bias1[2]);
	//随机生成bias2
	for(int i=0; i<NEU_NUM2; i++){
		bias2[i] = 2*(rand()/(double)(RAND_MAX)) - 1;
	}
	fprintf(stdout, "Bias2: %lf %lf %lf\n",bias2[0],bias2[1],bias2[2]);

	SAFE_CALL(cudaMalloc((void **) &gpu_mre, sizeof(double) * mre_num * KER_NUM));//maxpooling结果存入gpu_mre，分配显存
	SAFE_CALL(cudaMalloc((void **) &gpu_mre_index, sizeof(int) * mre_num * KER_NUM));//为maxpooling的最大值索引分配显存
	SAFE_CALL(cudaMalloc((void **) &gpu_omega1, sizeof(double) * ome_num1));//第一层网络的输入权重，分配显存
	SAFE_CALL(cudaMalloc((void **) &gpu_omega2, sizeof(double) * ome_num2));//输出层的权重，分配显存
	SAFE_CALL(cudaMalloc((void **) &gpu_F1, sizeof(double) * NEU_NUM1));//第一层网络的输出，分配显存
	SAFE_CALL(cudaMalloc((void **) &gpu_O2, sizeof(double) * NEU_NUM2));//输出层的结果
	SAFE_CALL(cudaMalloc((void **) &gpu_bias1, sizeof(double) * NEU_NUM1));//偏置值
	SAFE_CALL(cudaMalloc((void **) &gpu_bias2, sizeof(double) * NEU_NUM2));
	SAFE_CALL(cudaMemcpy(gpu_omega1, omega1, sizeof(double) * ome_num1, cudaMemcpyHostToDevice));//复制初始权重到GPU端
	SAFE_CALL(cudaMemcpy(gpu_omega2, omega2, sizeof(double) * ome_num2, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_bias1, bias1, sizeof(double) * NEU_NUM1, cudaMemcpyHostToDevice));//复制偏置值到显存
	SAFE_CALL(cudaMemcpy(gpu_bias2, bias2, sizeof(double) * NEU_NUM2, cudaMemcpyHostToDevice));

	double * mre = new double [mre_num * KER_NUM];//CPU端存放maxpooling结果
	//double * F1 = new double [NEU_NUM1];//CPU端存放第一层网络输出结果
	double * O2 = new double [NEU_NUM2];//CPU端存放输出层的结果
	double * lz = new double [NEU_NUM2];
	for(int i0=0;i0<1;i0++){
		//if (i0 % 100 == 0)
		//	fprintf(stdout,"The %dst iteration.\n",i0);
		for(int j=0; j<10000000; j++){
			int iter = 0;

			//卷积，每个线程负责一个卷积核和训练数据的卷积
			convol<<<1,KER_NUM,(NEIGHBOR+1)*z*sizeof(double)>>>(iter,i0,gpu_processed_train,gpu_kernel,gpu_re,3,3,z,re_size);
			cudaDeviceSynchronize();	
			//下采样，maxpooling方法，每个线程负责re的一列
			maxpooling<<<1,KER_NUM>>>(iter,gpu_re,gpu_mre,gpu_mre_index,re_size,mre_num);
			cudaDeviceSynchronize();

			//全连接层
			fullconnect<<<1,NEU_NUM1,mre_size * sizeof(double)>>>(iter,gpu_mre,gpu_omega1,gpu_bias1,gpu_F1,mre_size);
			cudaDeviceSynchronize();

			//输出层
			output<<<1,NEU_NUM2>>>(iter,gpu_F1,gpu_omega2,gpu_bias2,gpu_O2);
			cudaDeviceSynchronize();
			SAFE_CALL(cudaMemcpy(O2, gpu_O2, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
			cudaDeviceSynchronize();
			double loss = lossfunction(O2, processed_labels, i0);
			if(i0%100==0 && j%50==0)    fprintf(stdout,"loss:%lf \n",loss);
			if(loss < 0.001)
				break;

			//反向传播，输出层
			bp_output<<<1,NEU_NUM2>>>(iter,i0,gpu_processed_labels,gpu_O2,gpu_bias2,gpu_delta_La,gpu_delta_Lz);
			//反向传播，全连接层
			bp_fullconnect<<<1,NEU_NUM1>>>(iter,gpu_omega2,gpu_bias1,gpu_F1,gpu_delta_La,gpu_delta_Lz,gpu_delta_fa,gpu_delta_fz);
			//反向传播，maxpooling层
			bp_maxpooling<<<1,mre_size>>>(iter,mre_size,gpu_mre_index,gpu_omega1,gpu_mre,gpu_delta_fa,gpu_delta_fz,gpu_delta_ma,gpu_delta_2,gpu_delta_22);
			//反向传播，map到卷积层
			//bp_map_convol<<<1,mre_size>>>(iter,mre_num,gpu_mre_index,gpu_re,gpu_delta_2,gpu_delta_22);
			bp_update_kernel<<<1,KER_NUM,(NEIGHBOR+1)*z*sizeof(double)>>>(iter,i0,3,3,z,mre_num,re_size,gpu_mre_index,gpu_delta_22,gpu_processed_train,gpu_delta_kw,gpu_kernel);

			cudaDeviceSynchronize();
			SAFE_CALL(cudaMemcpy(bias2, gpu_bias2, sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
			/*if(j <5){
				fprintf(stdout,"bias2:\n ");
				for(int j=0; j<16; j++)
					fprintf(stdout," %lf\n",bias2[j]);
				fprintf(stdout,"\n");
			}*/
		}
	}
	int * mre_index = new int [mre_size];

	fprintf(stdout,"Training completed!\n");
	//cudaDeviceSynchronize();
	//SAFE_CALL(cudaMemcpy(re, gpu_re, sizeof(double) * re_size * KER_NUM, cudaMemcpyDeviceToHost));
	//SAFE_CALL(cudaMemcpy(mre,gpu_mre,sizeof(double) * mre_num * KER_NUM, cudaMemcpyDeviceToHost));
	//SAFE_CALL(cudaMemcpy(mre_index,gpu_mre_index,sizeof(int) * mre_size, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaMemcpy(O2,gpu_O2,sizeof(double) * NEU_NUM2, cudaMemcpyDeviceToHost));
	//SAFE_CALL(cudaMemcpy(omega1, gpu_omega1, sizeof(double) * ome_num1, cudaMemcpyDeviceToHost));
	//SAFE_CALL(cudaMemcpy(omega2, gpu_omega2, sizeof(double) * ome_num2, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	//fprintf(stdout,"result:%lf %lf %lf\n",re[0],re[5],re[10]);
	//fprintf(stdout,"result:%lf %lf %lf\n",re[1],re[6],re[11]);
	//fprintf(stdout,"result:%lf %lf %lf\n",re[2],re[7],re[12]);
	//fprintf(stdout,"result:%lf %lf %lf\n",re[3],re[8],re[13]);
	//fprintf(stdout,"result:%lf %lf %lf\n",re[4],re[9],re[14]);

	//fprintf(stdout,"mre:%lf %lf %lf\n",mre[0],mre[1],mre[2]);
	//fprintf(stdout,"mre_index:%d %d %d\n",mre_index[0],mre_index[1],mre_index[2]);

	//fprintf(stdout,"F1 Output:%lf %lf; %lf %lf\n",F1[0],F1[1],F1[98],F1[99]);
	fprintf(stdout,"O2 Output:%lf %lf; %lf %lf\n",O2[0],O2[1],O2[18],O2[19]);
	
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

	const mwSize  * dim;
	dim = mxGetDimensions(train);//获取trainset每维的元素个数

	start = clock();
	int te = training(trainset, trainlabels, dim[0], dim[1], dim[2]);
	end = clock();
	double usetime = double(end - start);
	fprintf(stdout, "Using time of preprocessing:%lfs\n",usetime/CLOCKS_PER_SEC);
	return 0;
}
