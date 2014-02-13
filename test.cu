#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
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
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

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


int main(void)
{
	srand((unsigned)(time(NULL)));
	int size = 5;
	int numl = 5;

	cusp::hyb_matrix<int, float, cusp::device_memory> ** A;
	cusp::array1d<float, cusp::device_memory> * in;
	cusp::array1d<float, cusp::device_memory> * out;	


	//allocate memory for weights
	A = new cusp::hyb_matrix<int, float, cusp::device_memory> * [numl];
	
	//initialize weights	
	for( int i = 0; i < numl; i++)
	{
		// initialize matrix
		cusp::array2d<float, cusp::host_memory> B(size, size);
		for(int j = 0; j < size; j++)
		{
			for( int k = 0; k < size; k++)
			{
				B(j, k) = (float)(rand())/(RAND_MAX/2.0) - 1.0;
			}
		}
	   
		A[i] = new cusp::hyb_matrix<int, float, cusp::device_memory>(B);
	}


	// initialize input vector
	cusp::array1d<float, cusp::device_memory>temp(size);
	
	for(int i = 0; i < size; i++)
	{
		temp[i] = (float)((rand())/(RAND_MAX/2.0) - 1.0);

	}	
	
	in = &temp;
	cusp::print(*in);

	// allocate output vector
	out = new cusp::array1d<float, cusp::device_memory>(size);	

	for( int i = 1; i < numl; i++)
	{
		// compute y = A * x
		cusp::multiply(*A[i], *in, *out);
		sigmoid(*out);

		cusp::print(*out);
		cusp::print(*A[i]);
		
		*in = *out;
	}

	// print y
	//cusp::print(*in);

	return 0;
}






