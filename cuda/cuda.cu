#include <stdio.h>
#include <iostream>

using namespace std;


#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
#define TILE_WIDTH 2

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

void multiply (int *matrixA, int *matrixB, int *matrixC, int matrixSize)
{
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
        printf("GPU time taken: %g ms\n", elapsedTime);

        //vratim obsah pamate GPU do CPU
        cudaMemcpy(matrixC, devMatrixC, overallSize, cudaMemcpyDeviceToHost);

        // zobrazim vysledok
        for (int i=0; i<matrixSize; i++)
                for (int j=0; j<matrixSize; j++)
                        if ((j+1) == matrixSize)
                                printf ("%d\n", matrixC[i* matrixSize + j]);
                        else
                                printf ("%d ", matrixC[i* matrixSize + j]);

        // uvolnime pamat na GPU
        cudaFree (devMatrixA);
        cudaFree (devMatrixB);
        cudaFree (devMatrixC);

        // uvolnenie eventov
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
}

void readInputFile (Matrix* &m1, Matrix* &m2, MatrixList* &ml)
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
	
	ml = new MatrixList(m1,m2); // create a new object - MatrixList, which includes both matrixes
}

int main (int argc, char ** argv) {

        cout << "ahoj" << endl;

        int a[4][4] = { { 1,2,3,4 }, { 5,6,7,8 }, { 9,10,11,12 }, {1,2,3,4} };
        int b[4][4] = { { 9,10,11,12 }, {1,2,3,4}, {5,6,7,8}, {1,2,3,4} };
        int c[4][4] = { { 0 } };
        int size = 4;
        multiply(*a,*b,*c,size);
}
