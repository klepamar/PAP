#include <omp.h>
#include <iostream>
#include <fstream>
#include "MatrixList.h"

using namespace std;

extern bool verbose; // use verbose variable defined in index.cpp
extern int block; // use block variable defined in index.cpp
extern int cpuNumber; // use variable defining no of active processors defined in index.cpp

inline int min (int a, int b) // define as inline function (= entire function definition will be inserted into the code instead of performing jumps to its memory location)
{
  return (a < b) ? a : b;
}

MatrixList::MatrixList (Matrix* m1, Matrix* m2)
{
	this->m1=m1; // or with constructor?
	this->m2=m2;
	this->mClassic = this->mClassicOptimised = this->mStrassen = NULL;
		
	// display input matrixes
//	m1->displayMatrix();
//	m2->displayMatrix();
}

MatrixList::~MatrixList()
{
	delete mClassic;
	delete mClassicOptimised;
	delete mStrassen;
}

bool MatrixList::isCorrectSize () const
{
	if (this->m1->getDimY() == this->m2->getDimX()) return true;
	return false;
}

void MatrixList::classic ()
{
	// here goes classical approach
	mClassic = new Matrix (m1->getDimX(),m2->getDimY());

	int i,j,k; // premenne cyklu

	start_timer=omp_get_wtime();
	
	omp_set_num_threads(cpuNumber); // nastavenie poctu spustenych vlakien	
	#pragma omp parallel for collapse (2) default(shared) private(i,j,k) schedule(static)
		for (i=0; i<m1->getDimX(); i++) // m1-riadok
			for (j=0; j<m2->getDimY(); j++) // m2-stlpec
				for (k=0; k<m1->getDimY(); k++) // m1-stlpec == m2-riadok
					this->mClassic->getMatrix()[i][j] += this->m1->getMatrix()[i][k] * this->m2->getMatrix()[k][j];
	end_timer=omp_get_wtime();	
	classic_length=end_timer-start_timer;
	
	mClassic->showMatrix();
//	mClassic->displayMatrix();
	if (verbose) cout << "Classical algorithm (" << cpuNumber << " threads) took " << classic_length << " seconds." << endl;
}

void MatrixList::classicOptimised()
{
	// classical approach with loop tiling mechanism
	mClassicOptimised = new Matrix (m1->getDimX(),m2->getDimY());
	
	int i,j,k,ii,jj,kk; // premenne cyklu
	
	start_timer=omp_get_wtime();

	omp_set_num_threads(cpuNumber); // nastavenie poctu spustenych vlakien
	#pragma omp parallel for collapse (2) default(shared) private(i,j,k,ii,jj,kk) schedule(static)
	for (ii=0; ii<m1->getDimX(); ii+=block)
		for (jj=0; jj<m2->getDimY(); jj+=block)
			for (kk=0; kk<m1->getDimY(); kk+=block)
				for (i=ii; i<min(ii + block, m1->getDimX()); i++) // loop tiling - perform multiplication on submatrixes 
					for (j=jj; j<min(jj + block, m2->getDimY()); j++) 
						for (k=kk; k<min(kk + block, m1->getDimY()); k+=2) // loop unrolling - perform two additions in one iteration and therefore decrease no of iterations by half
						{
							this->mClassicOptimised->getMatrix()[i][j] += this->m1->getMatrix()[i][k] * this->m2->getMatrix()[k][j];
							if ((k+1) != min(kk + block, m1->getDimY())) // do not access memory behind the last element at the end of the loop
								this->mClassicOptimised->getMatrix()[i][j] += this->m1->getMatrix()[i][k+1] * this->m2->getMatrix()[k+1][j];
						}
	end_timer=omp_get_wtime();
	optimised_length=end_timer-start_timer;
	
	mClassicOptimised->showMatrix();
//	mClassicOptimised->displayMatrix();
//	if (verbose) cout << "Classical algorithm with loop tiling took " << optimised_length << " seconds (" << (optimised_length/classic_length)*100 << " % of the original value)." << endl;
	if (verbose) cout << "Classical algorithm with loop tiling (" << cpuNumber << " threads) took " << (double)optimised_length << " seconds." << endl;
}

void MatrixList::add(Matrix *A, int A_off_X, int A_off_Y, Matrix *B, int B_off_X, int B_off_Y, Matrix *C, int C_off_X, int C_off_Y, int size){
//	#pragma omp parallel for collapse (2) default(shared) schedule(static)
	for (int i=0; i<size; i++)
		for (int j=0; j<size; j++)
			C->getMatrix()[i+C_off_X][j+C_off_Y] = A->getMatrix()[i+A_off_X][j+A_off_Y] + B->getMatrix()[i+B_off_X][j+B_off_Y];
}

void MatrixList::sub(Matrix *A, int A_off_X, int A_off_Y, Matrix *B, int B_off_X, int B_off_Y, Matrix *C, int C_off_X, int C_off_Y, int size){
//	#pragma omp parallel for collapse (2) default(shared) schedule(static)
	for (int i=0; i<size; i++)
		for (int j=0; j<size; j++)
			C->getMatrix()[i+C_off_X][j+C_off_Y] = A->getMatrix()[i+A_off_X][j+A_off_Y] - B->getMatrix()[i+B_off_X][j+B_off_Y];
}


void MatrixList::mul(Matrix *A, int A_off_X, int A_off_Y, Matrix *B, int B_off_X, int B_off_Y, Matrix *C, int C_off_X, int C_off_Y, int size){
//	#pragma omp parallel for collapse (2) default(shared) schedule(static)
		for (int i=0; i<size; i++)
			for (int j=0; j<size; j++)
				for (int k=0; k<size; k++)
					C->getMatrix()[i+C_off_X][j+C_off_Y] += A->getMatrix()[i+A_off_X][k+A_off_Y] * B->getMatrix()[k+B_off_X][j+B_off_Y];
}

void MatrixList::compute(Matrix *A, int A_off_X, int A_off_Y, Matrix *B, int B_off_X, int B_off_Y, Matrix *C, int C_off_X, int C_off_Y, int size){
	if (size <= 64) {
		this->mul(A,A_off_X,A_off_Y,B,B_off_X,B_off_Y,C,C_off_X,C_off_Y,size);
		return;
	}

	

	int halfSize = size / 2;
	Matrix * p1 = new Matrix(halfSize,halfSize);
	Matrix * p2 = new Matrix(halfSize,halfSize);
	Matrix * p3 = new Matrix(halfSize,halfSize);
	Matrix * p4 = new Matrix(halfSize,halfSize);
	Matrix * p5 = new Matrix(halfSize,halfSize);
	Matrix * p6 = new Matrix(halfSize,halfSize);
	Matrix * p7 = new Matrix(halfSize,halfSize);

	Matrix * aR1 = new Matrix(halfSize,halfSize);
	Matrix * bR1 = new Matrix(halfSize,halfSize);

	Matrix * aR2 = new Matrix(halfSize,halfSize);
	Matrix * bR2 = new Matrix(halfSize,halfSize);

	Matrix * aR3 = new Matrix(halfSize,halfSize);
	Matrix * bR3 = new Matrix(halfSize,halfSize);

	Matrix * aR4 = new Matrix(halfSize,halfSize);
	Matrix * bR4 = new Matrix(halfSize,halfSize);



//#pragma omp parallel
//{
#pragma omp task default (shared)
{
	this->add(A,A_off_X+0,A_off_Y+0,A,A_off_X+halfSize,A_off_Y+halfSize,aR1,0,0,halfSize);		//a11+a22
	this->add(B,B_off_X+0,B_off_Y+0,B,B_off_X+halfSize,B_off_Y+halfSize,bR1,0,0,halfSize);		//b11+b22
	this->compute(aR1,0,0,bR1,0,0,p1,0,0,halfSize);							//p1=a11+a22 * b11+b22

	this->add(A,A_off_X+halfSize,A_off_Y+0,A,A_off_X+halfSize,A_off_Y+halfSize,aR1,0,0,halfSize);	//a21+a22
	this->compute(aR1,0,0,B,B_off_X+0,B_off_Y+0,p2,0,0,halfSize);					//p2=a21+a22 * b11
}

#pragma omp task default (shared)
{
	this->sub(B,B_off_X+halfSize,B_off_Y+0,B,B_off_X+0,B_off_Y+0,bR2,0,0,halfSize);		//b21-b11
	this->compute(A,A_off_X+halfSize,A_off_Y+halfSize,bR2,0,0,p4,0,0,halfSize);			//p4=a22 * b21-b11

	this->sub(A,A_off_X+halfSize,A_off_Y+0,A,A_off_X+0,A_off_Y+0,aR2,0,0,halfSize);			//a21-a11
	this->add(B,B_off_X+0,B_off_Y+0,B,B_off_X+0,B_off_Y+halfSize,bR2,0,0,halfSize);			//b11+b12
	this->compute(aR2,0,0,bR2,0,0,p6,0,0,halfSize);							//p6=a21-a11 * b11+b22
}


#pragma omp task default (shared)
{
	this->sub(B,B_off_X+0,B_off_Y+halfSize,B,B_off_X+halfSize,B_off_Y+halfSize,bR3,0,0,halfSize);	//b12-b22
	this->compute(A,A_off_X+0,A_off_Y+0,bR3,0,0,p3,0,0,halfSize);					//p3=a11 * b12-b22

	this->sub(A,A_off_X+0,A_off_Y+halfSize,A,A_off_X+halfSize,A_off_Y+halfSize,aR3,0,0,halfSize);	//a12-a22
	this->add(B,B_off_X+halfSize,B_off_Y+0,B,B_off_X+halfSize,B_off_Y+halfSize,bR3,0,0,halfSize);	//b21+b22
	this->compute(aR3,0,0,bR3,0,0,p7,0,0,halfSize);							//p7=a12-a22 * b21+b22
}

	this->add(A,A_off_X+0,A_off_Y+0,A,A_off_X+0,A_off_Y+halfSize,aR4,0,0,halfSize);			//a11+a12
	this->compute(aR4,0,0,B,B_off_X+halfSize,B_off_Y+halfSize,p5,0,0,halfSize);			//p5=a11+a12 * b22


#pragma omp taskwait

#pragma omp task default (shared)
{
	this->add(p3,0,0,p5,0,0,C,C_off_X+0,C_off_Y+halfSize,halfSize);					//c12=p3+p5
}

#pragma omp task default (shared)
{
	this->add(p2,0,0,p4,0,0,C,C_off_X+halfSize,C_off_Y+0,halfSize);					//c21=p2+p4
}

#pragma omp task default (shared)
{
	this->add(p1,0,0,p4,0,0,aR1,0,0,halfSize);							//p1+p4
	this->add(aR1,0,0,p7,0,0,bR1,0,0,halfSize);							//p1+p4+p7
	this->sub(bR1,0,0,p5,0,0,C,C_off_X+0,C_off_Y+0,halfSize);					//c11=p1+p4+p7-p5
}

	this->add(p1,0,0,p3,0,0,aR2,0,0,halfSize);							//p1+p3
	this->add(aR2,0,0,p6,0,0,bR2,0,0,halfSize);							//p1+p3+p6
	this->sub(bR2,0,0,p2,0,0,C,C_off_X+halfSize,C_off_Y+halfSize,halfSize);				//c22=p1+p3+p6-p2

#pragma omp taskwait

	delete p1;
	delete p2;
	delete p3;
	delete p4;
	delete p5;
	delete p6;
	delete p7;
	delete aR1;
	delete bR1;
	delete aR2;
	delete bR2;
	delete aR3;
	delete bR3;
}


int MatrixList::nextPowerOf2(int number) const {
	return (pow(2,ceil(log2(number))));
}

int MatrixList::maxDim() const {
	if (this->m1->getDimX() >= this->m1->getDimY() and this->m1->getDimX() >= this->m2->getDimY())
		return (this->m1->getDimX());

	if (this->m1->getDimY() >= this->m1->getDimX() and this->m1->getDimY() >= this->m2->getDimY())
		return (this->m1->getDimY());

	if (this->m2->getDimY() >= this->m1->getDimY() and this->m2->getDimY() >= this->m1->getDimY())
		return (this->m2->getDimY());
}

void MatrixList::strassen () {
	omp_set_num_threads(cpuNumber); // nastavenie poctu spustenych vlakien

	this->mStrassen = new Matrix (m1->getDimX(),m2->getDimY());

	if (this->m1->getDimX() != this->m1->getDimY() or
	    this->m2->getDimX() != this->m2->getDimY() or
            this->m1->getDimX() != this->m2->getDimY() or
            this->nextPowerOf2(this->m1->getDimX()) != this->m1->getDimX()) {

		int newSize = this->nextPowerOf2(this->maxDim());

		Matrix * m1Resize = new Matrix(newSize,newSize);
		Matrix * m2Resize = new Matrix(newSize,newSize);
		Matrix * mStrassenResize = new Matrix(newSize,newSize);

		for (int i=0; i<m1->getDimX(); i++)
			for (int j=0; j<m1->getDimY(); j++)
				m1Resize->getMatrix()[i][j] = m1->getMatrix()[i][j];
		
		for (int i=0; i<m2->getDimX(); i++)
			for (int j=0; j<m2->getDimY(); j++)
				m2Resize->getMatrix()[i][j] = m2->getMatrix()[i][j];

		start_timer=omp_get_wtime();
#pragma omp parallel 
{
#pragma omp single
{
		this->compute(m1Resize,0,0,m2Resize,0,0,mStrassenResize,0,0,newSize);
}
}
		end_timer=omp_get_wtime();

		for (int i=0; i<this->mStrassen->getDimX(); i++)
			for (int j=0; j<this->mStrassen->getDimY(); j++)
				this->mStrassen->getMatrix()[i][j] = mStrassenResize->getMatrix()[i][j];

		delete m1Resize;
		delete m2Resize;
		delete mStrassenResize;
	}
	else {
		start_timer=omp_get_wtime();
#pragma omp parallel 
{
#pragma omp single
{
		this->compute(this->m1,0,0,this->m2,0,0,this->mStrassen,0,0,this->m1->getDimX());
}
}
		end_timer=omp_get_wtime();
	}
	
	this->mStrassen->showMatrix();
	optimised_length = end_timer - start_timer;
	if (verbose) cout << "Strassen algorithm (" << cpuNumber << " threads) took " << optimised_length << " seconds." << endl;

}


void MatrixList::cuda()
{
	cout << "hello world" << endl;
}
