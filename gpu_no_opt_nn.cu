#include <sys/param.h>
#include <sys/times.h>
#include <sys/types.h>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <new>
#include <string>

using namespace std;

__global__ void mv_kernel(float * in, float * A, float * out, int m, int n )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float sum = 0;
	for(int j = 0; j < m; j++)
	{
		sum = sum + (in[j] * A[(j*n)+i]);
	}
	out[i] = 1/(1+(powf(2.71828183,(-1*sum))));
}

class NeuralNet
{

//      no of layers in net including input layer
        int numl;

//      array of numl elements to store size of each layer
        int lsize;

//      sigmoid function
        float sigmoid(float in);

//	this variable holds the size of the data structure in bytes
	long size;

	//cuda vectors
	float **d_weights, *d_in, *d_out;


public:

        ~NeuralNet();

//      initializes and allocates memory
	NeuralNet(int nl, int lsz, int conn_prop);

//      feed forwards activations for one set of inputs
        long unsigned int ffwd();

//	gets the size of the data structure
	long unsigned int getsize();	

};

timespec diff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) 
	{
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} 
	else 
	{
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

long unsigned int NeuralNet::getsize()
{
	return size;
}	

// initializes and allocates memory
NeuralNet::NeuralNet(int nl, int lsz, int conn_prop)
{
	size = 0;
	numl=nl;
    	lsize = lsz;

    	//allocate memory for input of each neuron
    	cudaMalloc((void**)&d_in, lsize * sizeof(float));
	size = size + (lsize * sizeof(float));

    	//allocate memory for output of each neuron
    	cudaMalloc((void**)&d_out, lsize * sizeof(float));
	size = size + (lsize * sizeof(float));

	//allocate memory for weights

	d_weights = new float*[numl];

	for(int i = 0; i < numl; i++)
	{
		cudaMalloc((void**)&d_weights[i], lsize * lsize * sizeof(float));	
		size = size + (lsize * lsize * sizeof(float));
	}

	//seed and assign the input and weights
    	srand((unsigned)(time(NULL)));

	float * in = new float[lsize];

    	for(int i = 0; i < lsize; i++)
		in[i]= (float)(rand())/(RAND_MAX/2) - 1;

	cudaMemcpy(d_in, in, lsize * sizeof(float), cudaMemcpyHostToDevice);

	delete[] in;

	//seed and assign random numbers
	float * weights = new float[lsize * lsize];

    	for(int i = 0; i < numl; i++)
	{
		//assign zeroes to the matrix to be initialized
		for(int j = 0; j < lsize * lsize; j++) weights[j] = 0.0;
		for(int counter = 0; counter < (int)(lsize * lsize * (conn_prop / 100.0));)
		{
			int x = rand() % (lsize * lsize);
			if( weights[x] == 0.0)
			{
				weights[x] = (float)(rand())/(RAND_MAX/2) - 1;		
				counter++;
			}
		}
		cudaMemcpy(d_weights[i], weights, lsize * lsize * sizeof(float), cudaMemcpyHostToDevice);	
	}
	delete[] weights;

}


NeuralNet::~NeuralNet()
{
	for(int i = 0; i < numl; i++)
	{
		cudaFree(d_weights[i]);
	}	 
	cudaFree(d_in); 
	cudaFree(d_out);
    	cudaThreadExit();

}

long unsigned int NeuralNet::ffwd()
{
	
	timespec time1, time2;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
	
	for(int i = 1; i < numl; i++)
	{	
		mv_kernel<<<1, lsize>>>(d_in, d_weights[i], d_out, lsize, lsize);
		float *inter;
		inter = d_out;
		d_out = d_in;
		d_out = inter;		
	}

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

	return diff(time1,time2).tv_sec*1000000000 + diff(time1,time2).tv_nsec;
}


// A: m-by-n matrix, x : n elements vector, y : m elements vector, 
main(int argc, char* argv[])
{
	string str_layers, str_neurons, str_conn;

	string temp = argv[1];
	int layers = atoi(temp.data());
	temp = argv[2];
	int neurons_per_layer = atoi(temp.data());
	temp = argv[3];
	int conn_prop = atoi(temp.data());

	long unsigned int time_results[10];
	long unsigned int size_results[10];
	for(int i = 0; i < 10; i++)
	{
		NeuralNet *nn = new NeuralNet(layers, neurons_per_layer, conn_prop);
		time_results[i] = nn->ffwd();
		size_results[i] = nn->getsize();
		delete nn;
	}
	long unsigned int size_sum = 0;
	long unsigned int time_sum = 0;

	for(int i = 0; i < 10; i++)
	{
		time_sum = time_sum + time_results[i];	
		size_sum = size_sum + size_results[i];	
	}
	size_sum = size_sum / 10;
	time_sum = time_sum / 10;

	cout << layers << " " << neurons_per_layer << " " << conn_prop << " " << time_sum << " " << size_sum << " " << endl;
}