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
#include <string>

using namespace std;

class NeuralNet
{

//      output
        float *out;

//	input
	float *in;

//      3-D array to store weights for each neuron
        float ***weight;

//      no of layers in net including input layer
        int numl;

//      array of numl elements to store size of each layer
        int lsize;

//      sigmoid function
        float sigmoid(float in);

//	this variable holds the size of the data structure in bytes
	unsigned long int size;

public:

        ~NeuralNet();

//      initializes and allocates memory
	NeuralNet(int nl,int sz, int conn_prop);

//      feed forwards activations for one set of inputs
        unsigned long int ffwd();

//      returns i'th output of the net
        float Out(int i) const;

//	gets the size of the data structure
	unsigned long int getsize();	

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

float NeuralNet::sigmoid(float in)
{
	return (float)1/(1+(pow(2.71828183,(-1*in))));
}

unsigned long int NeuralNet::getsize()
{
	return size;
}	

// initializes and allocates memory
NeuralNet::NeuralNet(int nl, int sz, int conn_prop)
{
	size = 0;
	//set no of layers and their sizes
	numl=nl;
    	lsize = sz;

    	//    allocate memory for output of each neuron

    	out = new float[lsize];
	size = size + (lsize * sizeof(float));

    	in = new float[lsize];
	size = size + (lsize * sizeof(float));

    	//    allocate memory for weights

    	weight = new float**[numl];
	size = size + (numl * sizeof(float**));

    	for(int i = 1;i < numl; i++)
	{
        	weight[i]=new float*[lsize];
		size = size + (numl * sizeof(float*));
    	}

    	for(int i = 1;i < numl; i++)
	{
        	for(int j=0; j < lsize; j++)
		{
            		weight[i][j]=new float[lsize];
			size = size + (sizeof(float) * lsize);
            		
			for( int k = 0;k < lsize; k++)
			{
				weight[i][j][k]= 0.0;
			}
        	}
    	}

    	//    seed and assign random weights
    	srand((unsigned)(time(NULL)));
	
	for(int i = 1; i < numl; i++)
	{
		for( int counter = 0; counter < (int)(lsize * lsize * (conn_prop / 100.0)); )
		{
			int x = rand() % lsize;
			int y = rand() % lsize;
			if( weight[i][x][y] == 0.0)
			{
				weight[i][x][y] = (float)(rand())/(RAND_MAX/2) - 1;		
				counter++;
			}
		}
	}

}

NeuralNet::~NeuralNet()
{
	delete in;
	delete out;


    	//    allocate memory for weights

    	for(int i=1;i<numl;i++)
	{
        	for(int j=0;j<lsize;j++)
		{
            		delete weight[i][j];
		}
    	}

    	for(int i=1;i<numl;i++)
	{
        	delete weight[i];
    	}

	delete weight;


}

unsigned long int NeuralNet::ffwd()
{
	float sum;

 	//assign content to input layer
	srand((unsigned)(time(NULL)));

	for(int i=0;i < lsize;i++)
                in[i]=(float)(rand())/(RAND_MAX/2) - 1.0;

	timespec time1, time2;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

	// assign output(activation) value
	// to each neuron usng sigmoid func For each layer

        for(int i = 1; i < numl; i++ )
	{
		// For each neuron in current layer
                for(int j = 0; j < lsize; j++ )
		{
                        sum=0.0;
                        // For input from each neuron in preceding layer

                        for(int k = 0; k < lsize; k++)
			{
                                // Apply weight to inputs and add to sum
                                sum+= in[k]*weight[i][j][k];
                        }

                        // Apply sigmoid function
                        out[j]=sigmoid(sum);
		}
		float * inter;
		inter = in; 
		in = out;
		out = inter;
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

	return diff(time1,time2).tv_sec*1000000000 + diff(time1,time2).tv_nsec;
}

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