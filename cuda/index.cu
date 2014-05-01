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



/* DEFINE "GLOBAL" VARIABLES HERE */

char* filename; // where input matrixes are stored
bool verbose = false; // produce debugging messages with -v flag
int block = 100; // block size used for loop tiling in classical approach
int cpuNumber; // number of processesors active during OpenMP execution

/* END OF GLOBAL VARIABLES DEFINITION */

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

/* display help when invalid option has been used */
void displayHelp ()
{
	cout << "Usage:" << endl <<
			"\t-f FILE\tinput file with matrixes" << endl <<
			"\t-n NO_OF_CPUS\t number of active CPUs" << endl <<
			"\t-v\tverbose mode (optional)" << endl;
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
		else if (strcmp(argv[i],"-v") == 0) // -v...verbose mode
		{
			verbose = true;
		}
		else if (strcmp(argv[i],"-n") == 0) // -n...number of active CPUs
		{
			cpuNumber = atoi(argv[i+1]);
			i++;
		}
		else
		{
			displayHelp();
			throw "unknown arguments provided.";
		}
	}
}

static void HandleError(cudaError_t err, const char *file, int line)
{
        if (err != cudaSuccess)
        {
                printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
                exit(EXIT_FAILURE);
        }
}

__global__
void GPU_kernel1 (int * matrixA, int *matrixB, int *matrixC, int matrixSize)
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

__global__
void GPU_kernel2 (int * matrixA, int * matrixB, int * matrixC, int matrixSize) {
        __shared__ int As[TILE_WIDTH][TILE_WIDTH];
        __shared__ int Bs[TILE_WIDTH][TILE_WIDTH];

        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = by * TILE_WIDTH + ty;
        int col = bx * TILE_WIDTH + tx;
        float Cvalue = 0;

        for (int phase = 0; phase < matrixSize/TILE_WIDTH; phase ++) {
                As[ty][tx] = matrixA[row*matrixSize + (phase * TILE_WIDTH + tx)];
                Bs[ty][tx] = matrixB[col + (phase*TILE_WIDTH + ty)*matrixSize];

                __syncthreads();

                for (int k=0; k<TILE_WIDTH; k++)
                        Cvalue += As[ty][k] * Bs[k][tx];

                __syncthreads();
        }

        matrixC[row*matrixSize+col] = Cvalue;
}

void multiply (int *matrixA, int *matrixB, int *matrixC, int matrixSize) {
	// lokalne CUDA premenne
        cudaDeviceProp prop;
        cudaEvent_t start, stop;
        float elapsedTime;
        int whichDevice;

        // error handling
        HANDLE_ERROR(cudaGetDevice(&whichDevice));
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));

        // meranie casu
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // pointre do globalnej GPU pamate
        int *devMatrixA;
        int *devMatrixB;
        int *devMatrixC;

        // celkova velkost matice
        int overallSize = matrixSize * matrixSize * sizeof(int);

        // alokujem potrebne miesto na GPU
        cudaMalloc ((void **)&devMatrixA, overallSize);
        cudaMalloc ((void **)&devMatrixB, overallSize);
        cudaMalloc ((void **)&devMatrixC, overallSize);

        // skopirujem obsah pamate jednotlivych poli z MM do globalnej pamate GPU
        cudaMemcpy(devMatrixA, matrixA, overallSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devMatrixB, matrixB, overallSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devMatrixC, matrixC, overallSize, cudaMemcpyHostToDevice);

        // zaciatok merania casu
        cudaEventRecord(start, 0);

        // definicia velkosti mriezky & poctu vlakien v nej
        dim3 dimGrid(matrixSize/TILE_WIDTH, matrixSize/TILE_WIDTH);
        dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

        //spustim kernel
        GPU_kernel2<<<dimGrid,dimBlock>>>(devMatrixA, devMatrixB, devMatrixC, matrixSize);

        // koniec merania casu & vypocet casu straveneho na GPU
        cudaThreadSynchronize();
        cudaEventRecord(stop, 0) ;
        cudaEventSynchronize(stop) ;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        //vratim obsah pamate GPU do CPU
        cudaMemcpy(matrixC, devMatrixC, overallSize, cudaMemcpyDeviceToHost);

	//zobrazim vysledok
	printf("GPU time taken: %g ms\n", elapsedTime); 

        // uvolnime pamat na GPU
        cudaFree (devMatrixA);
        cudaFree (devMatrixB);
        cudaFree (devMatrixC);

        // uvolnenie eventov
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
//	m3->showMatrix();
	
	delete m1; // matrixes created in readInputFile method
	delete m2;
	delete m3; 
	
	
	return (EXIT_SUCCESS);
}
