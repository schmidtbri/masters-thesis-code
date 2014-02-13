#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <iostream>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <cusp/hyb_matrix.h>
#include <sys/param.h>
#include <sys/times.h>
#include <sys/types.h>
#include <ctime>
#include <sys/time.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>



using namespace std;

struct functor
{
	__host__ __device__ float operator()(const float& x) const 
	{ 
		return (1 / (1 + ( powf ( 2.71828183, (-1 * x ) ) ) ) );
        }
};

void sigmoid(cusp::array1d<float, cusp::device_memory>& X)
{
    thrust::transform(X.begin(), X.end(), X.begin(), functor());
}

class NeuralNet
{

//      no of layers in net including input layer
        int numl;

//      array of numl elements to store size of each layer
        int lsize;

//	this variable holds the size of the data structure in bytes
	long size;

	cusp::hyb_matrix<int, float, cusp::device_memory> ** A;
	cusp::array1d<float, cusp::device_memory> * in;
	cusp::array1d<float, cusp::device_memory> * out;	


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
	srand((unsigned)(time(NULL)));

	size = 0;
	numl=nl;
    	lsize = lsz;

	//allocate memory for weights
	A = new cusp::hyb_matrix<int, float, cusp::device_memory> * [numl];

	//initialize weights
	
	for( int i = 0; i < numl; i++)
	{
		// initialize matrix
		cusp::array2d<float, cusp::host_memory> B(lsize, lsize);
		
		//initialize the whole matrix to zeroes
		for(int j = 0; j < lsize; j++){
			for( int k = 0; k < lsize; k++){
				B(j,k) = 0.0;}}
		
		for(int counter = 0; counter < (int)(lsize * lsize * (conn_prop / 100.0));)
		{
			int x = rand() % lsize;
			int y = rand() % lsize;

			if( B(x, y) == 0.0)
			{
				B(x, y) = (float)(rand())/(RAND_MAX/2) - 1.0;		
				counter++;
			}
		}
		A[i] = new cusp::hyb_matrix<int, float, cusp::device_memory>(B);
		size = size +  (A[i]->ell.values.values.size() * sizeof(float)) + (A[i]->ell.column_indices.values.size() * sizeof(float));
		size = size + (A[i]->coo.num_entries * (sizeof(float) + (2 * sizeof(float))));
	}

	// initialize input vector
	cusp::array1d<float, cusp::device_memory>temp(lsize);
	size = size + (lsize * sizeof(float));
	
	for(int i = 0; i < lsize; i++)
		temp[i] = (float)((rand())/(RAND_MAX/2.0) - 1.0);
		
	in = &temp; 

	// allocate output vector
	out = new cusp::array1d<float, cusp::device_memory>(lsize);	
	size = size + (lsize * sizeof(float));
}


NeuralNet::~NeuralNet()
{
//	for(int i = 0; i < numl; i++)
//	{
//		delete A[i];	
//	}	 
//	delete out;
//	delete in;
}

long unsigned int NeuralNet::ffwd()
{
	timespec time1, time2;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
	for( int i = 1; i < numl; i++)
	{
		// compute y = A * x
		cusp::multiply(*A[i], *in, *out);
		sigmoid(*out);

		cusp::array1d<float, cusp::device_memory> * inter;
		inter = in;
		in = out;
		out = inter;
				
		
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