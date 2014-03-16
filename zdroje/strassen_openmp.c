/* 
OpenMP version of the Strassen Matrix Multiplication Algorithm 
James Mwaura, Honghao Tian

Compile options
	gcc -lrt -lm -fompenmp
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>

//matrix dimensions
/* DIM_N defines matix size, threads sets the openmp thread count      * 
 * threshold sets the lower limit for the strassen recursion algorithm *
 * chunk defines the openmp chunking size                              */
#define DIM_N 8192
#define threads 1
#define threshold 1024
int chunk=10;

//other stuff
double sum, snorm;
double *d;

//matrices
double **A, **B, **C;

//Prototypes
void strassenMultMatrix(double**,double**,double**,int);
void normalMultMatrix(double**, double**, double**, int);
void catMatrix(double**, double**,double**,double**,double**, int);
void splitMatrix(double**, double**,double**,double**,double**, int);
void subMatrices(double**, double**, double**, int);
void addMatrices(double**, double**, double**, int);

//MAIN
int main (int argc, char *argv[]){
  long stime, ntime;
  struct timespec start, finish;
  int i,j,k;
  
  A = (double**) malloc(sizeof(double)*DIM_N);
  for (i = 0; i< DIM_N; i++){
    A[i] = (double*) malloc(sizeof(double)*DIM_N);
  }
  B = (double**) malloc(sizeof(double)*DIM_N);
  for (i = 0; i< DIM_N; i++){
    B[i] = (double*) malloc(sizeof(double)*DIM_N);
  }
  C = (double**) malloc(sizeof(double)*DIM_N);
  for (i = 0; i< DIM_N; i++){
    C[i] = (double*) malloc(sizeof(double)*DIM_N);
  }
  
  d = (double*) malloc(sizeof(double)*DIM_N);
  sum = 0.0; snorm = 0.0;
  for (i = 0; i < DIM_N; i++){
    d[i] = drand48();
    snorm += d[i]*d[i];
  }
  for (i = 0; i < DIM_N; i++){
    d[i] = d[i]/sqrt(snorm);
    sum += d[i]*d[i];
  }

  for (i = 0; i < DIM_N; i++)
    for (j = 0; j < DIM_N; j++)
    A[i][j] =  -2*d[i]*d[j];

  for (i = 0; i < DIM_N; i++)
    A[i][i] = 1.0 + A[i][i];

  for (j = 0; j < DIM_N; j++)
    for (k = 0; k < DIM_N; k++)
    B[j][k] = -2*d[j]*d[k];

  for (j = 0; j < DIM_N; j++)
    B[j][j] = 1.0 + B[j][j];

  printf("Num Threads = %d\n",threads);
  //start timer
  clock_gettime(CLOCK_REALTIME,&start);
  
  //Strassen Multiplication
  strassenMultMatrix(A,B,C,DIM_N);

  //stop timer
  clock_gettime(CLOCK_REALTIME,&finish);
  
  //print out the result
  /*for (i=0; i<DIM_N; i++){
    for (j=0; j<DIM_N; j++)
      printf("%lf ",C[i][j]);
    printf("\n");
  }*/

  //calculate time taken
  ntime = finish.tv_nsec - start.tv_nsec;
  stime = (long) finish.tv_sec - (long) start.tv_sec;
  printf("Strassen Time taken = %lf \n", (double) stime + ((double) ntime)/1e9);
  
  
  /********Triple Loop Multiplication, with OpenMP, for Comparison**********/
  //start timer
  clock_gettime(CLOCK_REALTIME,&start);
  
  #pragma omp parallel shared(A,B,C,chunk) private(i,j,k) num_threads(threads)
	{
	  //multiplication process
    #pragma omp for schedule(dynamic) nowait
	    for (j = 0; j < DIM_N; j++){
		    for (i = 0; i < DIM_N; i++){
		      C[i][j] = 0.0;
			    for (k = 0; k < DIM_N; k++)
				    C[i][j] += A[i][k] * B[k][j];
				}
		  }
	}
  //normalMultMatrix(A,B,C,DIM_N);

  //stop timer
  clock_gettime(CLOCK_REALTIME,&finish);
  
  /*for (i=0; i<DIM_N; i++){
    for (j=0; j<DIM_N; j++)
      printf("%lf ",C[i][j]);
    printf("\n");
  }*/

  //calculate time taken
  ntime = finish.tv_nsec - start.tv_nsec;
  stime = (long) finish.tv_sec - (long) start.tv_sec;
  printf("Non-Strassen Time taken = %lf \n", (double) stime + ((double) ntime)/1e9);
}
/*****************************************************************************
 * Note that all following functions only deal with square matrices          *
 * where N is divisible by 2.                                                *
 *****************************************************************************/
void addMatrices(double **x, double **y, double **z, int size){
//performs a matrix addition operation, z=x+y
	int i,j;
	#pragma omp parallel shared(x,y,z,size,chunk) private(i,j) num_threads(threads) 
	{
     #pragma omp for schedule(dynamic,chunk) nowait
	      for (i = 0; i < size; i++)
		      for (j = 0; j < size; j++)
			      z[i][j] = x[i][j] + y[i][j];  
	}
}

void subMatrices(double **x, double **y, double **z, int size){
//performs a matrix subtraction operation, z=x-y
	int i,j;
	#pragma omp parallel shared(x,y,z,size,chunk) private(i,j) num_threads(threads)
	{
     #pragma omp for schedule(dynamic,chunk) nowait
	      for (i = 0; i < size; i++)
		      for (j = 0; j < size; j++)
			      z[i][j] = x[i][j] - y[i][j];
	}
}

void splitMatrix(double **a, double**a11,double **a12,double **a21,double **a22, int size){
//takes a matrix a adn splits it into its 4 quadrants.
	int i,j,x,y;
	int newsize = (int)size/2;
	x=0; y=0;
	#pragma omp parallel shared(a,a11,a12,a21,a22,x,y,newsize,chunk) private(i,j) num_threads(threads)
	{
    #pragma omp for schedule(dynamic,chunk) nowait
      for (i = x; i < newsize+x; i++)
	      for (j = y; j < newsize+y; j++)
		      a11[i-x][j-y] = a[i][j];
	}
	x=newsize; y=0;
	#pragma omp parallel shared(a,a11,a12,a21,a22,x,y,newsize,chunk) private(i,j) num_threads(threads)
	{
     #pragma omp for schedule(dynamic,chunk) nowait
	      for (i = x; i < newsize+x; i++)
		      for (j = y; j < newsize+y; j++)
			      a12[i-x][j-y] = a[i][j];
	}
	x=0; y=newsize;
	#pragma omp parallel shared(a,a11,a12,a21,a22,x,y,newsize,chunk) private(i,j) num_threads(threads)
	{
    #pragma omp for schedule(dynamic,chunk) nowait
      for (i = x; i < newsize+x; i++)
        for (j = y; j < newsize+y; j++)
          a21[i-x][j-y] = a[i][j];
	}
	x=newsize; y=newsize;
	#pragma omp parallel shared(a,a11,a12,a21,a22,x,y,newsize,chunk) private(i,j) num_threads(threads)
	{
    #pragma omp for schedule(dynamic,chunk) nowait
      for (i = x; i < newsize+x; i++)
        for (j = y; j < newsize+y; j++)
          a22[i-x][j-y] = a[i][j];
	}
}

void catMatrix(double **a, double **a11,double **a12,double **a21,double **a22, int size){
//does the inverse of the splitMatrix function
	int i,j,x,y;
	int oldsize = (int)size/2;
  x=0; y=0;
  #pragma omp parallel shared(a,a11,a12,a21,a22,x,y,oldsize,chunk) private(i,j) num_threads(threads)
  {
    #pragma omp for schedule(dynamic,chunk) nowait
      for (i = x; i < oldsize+x; i++)
        for (j = y; j < oldsize+y; j++)
          a[i][j] = a11[i-x][j-y];
	}
	x=oldsize; y=0;
	#pragma omp parallel shared(a,a11,a12,a21,a22,x,y,oldsize,chunk) private(i,j) num_threads(threads)
	{
    #pragma omp for schedule(dynamic,chunk) nowait
      for (i = x; i < oldsize+x; i++)
        for (j = y; j < oldsize+y; j++)
          a[i][j] = a12[i-x][j-y];
	}
	x=0; y=oldsize;
	#pragma omp parallel shared(a,a11,a12,a21,a22,x,y,oldsize,chunk) private(i,j) num_threads(threads)
	{
    #pragma omp for schedule(dynamic,chunk) nowait
      for (i = x; i < oldsize+x; i++)
        for (j = y; j < oldsize+y; j++)
          a[i][j] = a21[i-x][j-y];
	}
	x=oldsize; y=oldsize;
	#pragma omp parallel shared(a,a11,a12,a21,a22,x,y,oldsize,chunk) private(i,j) num_threads(threads)
	{
    #pragma omp for schedule(dynamic,chunk) nowait
      for (i = x; i < oldsize+x; i++)
        for (j = y; j < oldsize+y; j++)
          a[i][j] = a22[i-x][j-y];
	}
}

void normalMultMatrix(double **x, double **y, double **z, int size){
//multiplys two matrices: z=x*y
	int i,j,k;
	
	#pragma omp parallel shared(A,B,C,chunk) private(i,j,k) num_threads(threads)
	{
	  //multiplication process
    #pragma omp for schedule(dynamic) nowait
	    for (j = 0; j < size; j++){
		    for (i = 0; i < size; i++){
		      z[i][j] = 0.0;
			    for (k = 0; k < size; k++)
				    z[i][j] += x[i][k] * y[k][j];
				}
      }
	}
}

void strassenMultMatrix(double **a,double **b,double **c,int size){
//Performs a Strassen matrix multiply operation
//This does miracles, and is recursive
//To perform a miracle, it first performs a miracle
  double **a11, **a22, **a12, **a21;
  double **b11, **b22, **b12, **b21;
  double **m1, **m2, **m3, **m4, **m5, **m6, **m7; 
  double **t1, **t2, **t3, **t4, **t5, **t6, **t7, **t8, **t9, **t10;
  int newsize = (int)size/2;
  int i;
  if (size > threshold) {
    //Allocate memory....this could get expensive pretty quickly
    a11 = (double**) malloc(sizeof(double)*newsize);
    a12 = (double**) malloc(sizeof(double)*newsize);
    a21 = (double**) malloc(sizeof(double)*newsize);
    a22 = (double**) malloc(sizeof(double)*newsize);
    b11 = (double**) malloc(sizeof(double)*newsize);
    b12 = (double**) malloc(sizeof(double)*newsize);
    b21 = (double**) malloc(sizeof(double)*newsize);
    b22 = (double**) malloc(sizeof(double)*newsize);
    m1 = (double**) malloc(sizeof(double)*newsize);
    m2 = (double**) malloc(sizeof(double)*newsize);
    m3 = (double**) malloc(sizeof(double)*newsize);
    m4 = (double**) malloc(sizeof(double)*newsize);
    m5 = (double**) malloc(sizeof(double)*newsize);
    m6 = (double**) malloc(sizeof(double)*newsize);
    m7 = (double**) malloc(sizeof(double)*newsize);
    t1 = (double**) malloc(sizeof(double)*newsize);
    t2 = (double**) malloc(sizeof(double)*newsize);
    t3 = (double**) malloc(sizeof(double)*newsize);
    t4 = (double**) malloc(sizeof(double)*newsize);
    t5 = (double**) malloc(sizeof(double)*newsize);
    t6 = (double**) malloc(sizeof(double)*newsize);
    t7 = (double**) malloc(sizeof(double)*newsize);
    t8 = (double**) malloc(sizeof(double)*newsize);
    t9 = (double**) malloc(sizeof(double)*newsize);
    t10 = (double**) malloc(sizeof(double)*newsize);
    
    for (i=0; i < newsize; i++){
      a11[i] = (double*) malloc(sizeof(double)*newsize);
      a12[i] = (double*) malloc(sizeof(double)*newsize);
      a21[i] = (double*) malloc(sizeof(double)*newsize);
      a22[i] = (double*) malloc(sizeof(double)*newsize);
      b11[i] = (double*) malloc(sizeof(double)*newsize); 
      b12[i] = (double*) malloc(sizeof(double)*newsize); 
      b21[i] = (double*) malloc(sizeof(double)*newsize);
      b22[i] = (double*) malloc(sizeof(double)*newsize);    
      m1[i] = (double*) malloc(sizeof(double)*newsize);
      m2[i] = (double*) malloc(sizeof(double)*newsize);
      m3[i] = (double*) malloc(sizeof(double)*newsize);
      m4[i] = (double*) malloc(sizeof(double)*newsize);
      m5[i] = (double*) malloc(sizeof(double)*newsize);
      m6[i] = (double*) malloc(sizeof(double)*newsize);
      m7[i] = (double*) malloc(sizeof(double)*newsize);
      t1[i] = (double*) malloc(sizeof(double)*newsize);
      t2[i] = (double*) malloc(sizeof(double)*newsize);
      t3[i] = (double*) malloc(sizeof(double)*newsize);
      t4[i] = (double*) malloc(sizeof(double)*newsize);
      t5[i] = (double*) malloc(sizeof(double)*newsize);
      t6[i] = (double*) malloc(sizeof(double)*newsize);
      t7[i] = (double*) malloc(sizeof(double)*newsize);
      t8[i] = (double*) malloc(sizeof(double)*newsize);
      t9[i] = (double*) malloc(sizeof(double)*newsize);
      t10[i] = (double*) malloc(sizeof(double)*newsize);
    }

    splitMatrix(a,a11,a12,a21,a22,size);
    splitMatrix(b,b11,b12,b21,b22,size);
    
    addMatrices(a11,a22,t1,newsize);
    addMatrices(a21,a22,t2,newsize);
    addMatrices(a11,a12,t3,newsize);
    subMatrices(a21,a11,t4,newsize);
    subMatrices(a12,a22,t5,newsize);
    addMatrices(b11,b22,t6,newsize);
    subMatrices(b12,b22,t7,newsize);
    subMatrices(b21,b11,t8,newsize);
    addMatrices(b11,b12,t9,newsize);
    addMatrices(b21,b22,t10,newsize);
    
    strassenMultMatrix(t1,t6,m1,newsize);
    strassenMultMatrix(t2,b11,m2,newsize);
    strassenMultMatrix(a11,t7,m3,newsize);
    strassenMultMatrix(a22,t8,m4,newsize);
    strassenMultMatrix(t3,b22,m5,newsize);
    strassenMultMatrix(t4,t9,m6,newsize);
    strassenMultMatrix(t5,t10,m7,newsize);
    
    addMatrices(m1,m4,a11,newsize);
    subMatrices(m5,m7,a12,newsize);
    addMatrices(m3,m1,a21,newsize);
    subMatrices(m2,m6,a22,newsize);
    subMatrices(a11,a12,b11,newsize);
    addMatrices(m3,m5,b12,newsize);
    addMatrices(m2,m4,b21,newsize);
    subMatrices(a21,a22,b22,newsize);
    
    catMatrix(c,b11,b12,b21,b22,size);
    free(a11);free(a12);free(a21);free(a22);
    free(b11);free(b12);free(b21);free(b22);
    free(t1);free(t2);free(t3);free(t4);free(t5);free(t6);free(t7);free(t8);free(t9);free(t10);
    free(m1);free(m2);free(m3);free(m4);free(m5);free(m6);free(m7);
  }
  else {
    normalMultMatrix(a,b,c,size);
  }
}

