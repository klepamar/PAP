#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include "Matrix.h"

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
#define TILE_WIDTH 16

using namespace std;

/* global variables definition */
char* filename;

/* CUDA error handling */
static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
    {
		printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line );
        exit(EXIT_FAILURE);
    }
}

/* read source data from input file & initialise matrixes */
void readInputFile (Matrix* &m1, Matrix* &m2)
{
	int x,y; // dimensions
	string dummy;
	ifstream in;
	
	in.open(filename); // open file
	if (!in.is_open()) throw "could not open file.";

	in >> x >> y; // read the dimensions of m1
	getline(in,dummy); // get rid of new line character
	m1 = new Matrix(x,y);
	m1->fillMatrix(in); // fill in m1
	
	in >> x >> y; // the same for m2
	getline(in,dummy);
	m2 = new Matrix(x,y);
	m2->fillMatrix(in);
	
	in.close(); // end up processing file
}

/* verify that matrixes are square and matrix C can be computed */
void verifyMatrixes (Matrix* &m1, Matrix* &m2)
{
	if (m1->getDimX() != m1->getDimY())
		throw "matrix 1 must be square.";
	if (m2->getDimX() != m2->getDimY())
		throw "matrix 2 must be square.";
	if (m1->getDimY() != m2->getDimX())
		throw "resultant matrix cannot be computed from input matrixes.";
}

/* display help when invalid option has been used */
void displayHelp ()
{
	cout << "Usage:" << endl <<
			"\t-f FILE\tinput file with matrixes" << endl;
}

/* determine input file and possibly other arguments */
void processArguments (int argc, char** argv)
{
	if (argc == 1) // no arguments provided
	{
		displayHelp();
		throw "no arguments provided.";
	}
	for (int i=1; i<argc; i++) // first argument argv[0] is executable
	{
		if (strcmp(argv[i],"-f") == 0) // -f...input file
		{
			if ((i+1) == argc) throw "no input file provided.";
			filename = argv[i+1];
			i++; // next "i" is already resolved
		}
		else
		{
			displayHelp();
			throw "unknown arguments provided.";
		}
	}
}
/* very simple kernel used when matrixSize < TILE_WIDTH */
__global__
void GPU_kernelSmall (int * matrixA, int * matrixB, int * matrixC, int matrixSize)
{
	// cislo bloku, cislo vlakna
	int i = blockIdx.x;
	int j = threadIdx.x;
	int result;
	
	for (int k=0; k<matrixSize; k++)
	{
		result += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
	}
	matrixC[i * matrixSize + j] = result;
}


/* simple kernel with extensive usage of global GPU memory */
__global__
void GPU_kernel1 (int * matrixA, int * matrixB, int * matrixC, int matrixSize)
{
        // cislo bloku, cislo vlakna
        int i = blockIdx.y*TILE_WIDTH + threadIdx.y;
        int j = blockIdx.x*TILE_WIDTH + threadIdx.x;
        int result;

        for (int k=0; k<matrixSize; k++)
        {
                result += matrixA[i * matrixSize + k] * matrixB[k * matrixSize + j];
        }
        matrixC[i * matrixSize + j] = result;
}

/* more advanced kernel with copying of submatrixes from global GPU memory to shared memory */
__global__
void GPU_kernel2 (int * matrixA, int * matrixB, int * matrixC, int matrixSize) 
{
	// shared memory allocated for submatrixes
        __shared__ int As[TILE_WIDTH][TILE_WIDTH];
        __shared__ int Bs[TILE_WIDTH][TILE_WIDTH];

        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
	int subMatrixSize = matrixSize/TILE_WIDTH;

	// element to be calculated
        int row = by * TILE_WIDTH + ty;
        int col = bx * TILE_WIDTH + tx;
        int Cvalue = 0;

	// for all required blocks
        for (int temp=0; temp<subMatrixSize; temp++) 
        {
		// read from global memory & store in shared memory
	    	As[ty][tx] = matrixA[row * matrixSize + (temp * TILE_WIDTH + tx)];
		Bs[ty][tx] = matrixB[col + (temp * TILE_WIDTH + ty) * matrixSize];
            	
		// wait for reading from global memory
		__syncthreads();
            	
		// calculate using values stored in shared memory
		for (int k=0; k<TILE_WIDTH; k++)
            	{
			Cvalue += As[ty][k] * Bs[k][tx];
		}
            	
		// wait for calculation before proceeding to a new submatrix(=block)
		__syncthreads();
        }

	// after all blocks finished, copy value back into global GPU memory
        matrixC[row*matrixSize+col] = Cvalue;
}

/* run kernel GPU from this function, prepare required CUDA structures (for time measurement) */
void multiply (int *matrixA, int *matrixB, int *matrixC, int matrixSize) {
	
		// local CUDA variables
        cudaDeviceProp prop;
        cudaEvent_t start, stop;
        float elapsedTime;
        int whichDevice;

        // error handling
        HANDLE_ERROR(cudaGetDevice(&whichDevice));
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));

        // used for time measurement
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // global GPU pointers
        int *devMatrixA;
        int *devMatrixB;
        int *devMatrixC;

        // total size of matrix in memory
        int overallSize = matrixSize * matrixSize * sizeof(int);

        // allocation of required space on global GPU memory
        cudaMalloc ((void **)&devMatrixA, overallSize);
        cudaMalloc ((void **)&devMatrixB, overallSize);
        cudaMalloc ((void **)&devMatrixC, overallSize);

        // copy variables from MM to global GPU memory
        cudaMemcpy(devMatrixA, matrixA, overallSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devMatrixB, matrixB, overallSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devMatrixC, matrixC, overallSize, cudaMemcpyHostToDevice);

        // time measurement - start
        cudaEventRecord(start, 0);

        // specify block size & number of threads in respective dimensions
        dim3 dimGrid(matrixSize/TILE_WIDTH, matrixSize/TILE_WIDTH);
        dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

        // run kernel
        if (matrixSize < TILE_WIDTH)
		GPU_kernelSmall<<<matrixSize, matrixSize>>>(devMatrixA, devMatrixB, devMatrixC, matrixSize);
	else
		GPU_kernel2<<<dimGrid,dimBlock>>>(devMatrixA, devMatrixB, devMatrixC, matrixSize);

        // time measurmenet - end; calculate overall computation time
        cudaThreadSynchronize();
        cudaEventRecord(stop, 0) ;
        cudaEventSynchronize(stop) ;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        // copy variables from global GPU back to MM
        cudaMemcpy(matrixC, devMatrixC, overallSize, cudaMemcpyDeviceToHost);

		// display elapsed time
		printf("GPU time taken: %g ms\n", elapsedTime); 

        // global GPU variables deallocation
        cudaFree (devMatrixA);
        cudaFree (devMatrixB);
        cudaFree (devMatrixC);

        // CUDA events deallocation
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
}

int main(int argc, char** argv)
{	
	Matrix* m1=NULL;
	Matrix* m2=NULL;
	try
	{
		processArguments(argc,argv);
		readInputFile(m1,m2);
		verifyMatrixes(m1,m2);
	}
	catch (const char* exception)
	{
		cout << "Exception: " << exception << endl;
		delete m1; // matrixes created in readInputFile method
		delete m2;
		exit (EXIT_FAILURE);
	}
	
	Matrix * m3 = new Matrix(m1->getDimX(),m2->getDimY());
	
	multiply(m1->getMatrix(),m2->getMatrix(),m3->getMatrix(),m1->getDimX());
	//m1->showMatrix();
	//cout << endl;
	//m2->showMatrix();
	//cout << endl;
	m3->showMatrix();
	
	delete m1; // matrixes created in readInputFile method
	delete m2;
	delete m3; 
		
	return (EXIT_SUCCESS);
}
