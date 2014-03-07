#include <omp.h>
#include <iostream>
#include <fstream>
#include "MatrixList.h"

using namespace std;

extern bool verbose; // use verbose variable defined in index.cpp
extern int block; // use block variable defined in index.cpp

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
	m1->displayMatrix();
	m2->displayMatrix();
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

	start_timer=omp_get_wtime();

	for (int i=0; i<m1->getDimX(); i++) // m1-riadok
		for (int j=0; j<m2->getDimY(); j++) // m2-stlpec
			for (int k=0; k<m1->getDimY(); k++) // m1-stlpec == m2-riadok
				this->mClassic->getMatrix()[i][j] += this->m1->getMatrix()[i][k] * this->m2->getMatrix()[k][j];
				
	end_timer=omp_get_wtime();	
	classic_length=end_timer-start_timer;
	
	mClassic->displayMatrix();
	if (verbose) cout << "Classical algorithm took " << classic_length << " seconds." << endl;
}

void MatrixList::classicOptimised()
{
	// classical approach with loop tiling mechanism
	mClassicOptimised = new Matrix (m1->getDimX(),m2->getDimY());
	start_timer=omp_get_wtime();

	for (int ii=0; ii<m1->getDimX(); ii+=block)
		for (int jj=0; jj<m2->getDimY(); jj+=block)
			for (int kk=0; kk<m1->getDimY(); kk+=block)
				for (int i=ii; i<min(ii + block, m1->getDimX()); i++) // loop tiling - perform multiplication on submatrixes 
					for (int j=jj; j<min(jj + block, m2->getDimY()); j++) 
						for (int k=kk; k<min(kk + block, m1->getDimY()); k+=2) // loop unrolling - perform two additions in one iteration and therefore decrease no of iterations by half
						{
							this->mClassicOptimised->getMatrix()[i][j] += this->m1->getMatrix()[i][k] * this->m2->getMatrix()[k][j];
							if ((k+1) != min(kk + block, m1->getDimY())) // do not access memory behind the last element at the end of the loop
								this->mClassicOptimised->getMatrix()[i][j] += this->m1->getMatrix()[i][k+1] * this->m2->getMatrix()[k+1][j];
						}
	end_timer=omp_get_wtime();
	optimised_length=end_timer-start_timer;
	
	mClassicOptimised->displayMatrix();
	if (verbose) cout << "Classical algorithm with loop tiling took " << optimised_length << " seconds (" << (optimised_length/classic_length)*100 << " % of the original value)." << endl;
}

void MatrixList::add(Matrix *A, Matrix *B, Matrix *C, int size){
	for (int i=0; i<size; i++)
		for (int j=0; j<size; j++)
			C->getMatrix()[i][j] = A->getMatrix()[i][j] + B->getMatrix()[i][j];
}

void MatrixList::sub(Matrix *A, Matrix *B, Matrix *C, int size){
	for (int i=0; i<size; i++)
		for (int j=0; j<size; j++)
			C->getMatrix()[i][j] = A->getMatrix()[i][j] - B->getMatrix()[i][j];
}


void MatrixList::mul(Matrix *A, Matrix *B, Matrix *C, int size){
	for (int i=0; i<size; i++)
		for (int j=0; j<size; j++)
			for (int k=0; k<size; k++)
				C->getMatrix()[i][k] += A->getMatrix()[i][j] * B->getMatrix()[j][k];
}

void MatrixList::compute(Matrix *A, Matrix *B, Matrix *C, int size){
	if (size <= 1) {
		this->mul(A,B,C,size);
		return;
	}

	int halfSize = size / 2;
	Matrix * a11 = new Matrix(halfSize,halfSize);
	Matrix * a12 = new Matrix(halfSize,halfSize);
	Matrix * a21 = new Matrix(halfSize,halfSize);
	Matrix * a22 = new Matrix(halfSize,halfSize);
	Matrix * b11 = new Matrix(halfSize,halfSize);
	Matrix * b12 = new Matrix(halfSize,halfSize);
	Matrix * b21 = new Matrix(halfSize,halfSize);
	Matrix * b22 = new Matrix(halfSize,halfSize);
	Matrix * c11 = new Matrix(halfSize,halfSize);
	Matrix * c12 = new Matrix(halfSize,halfSize);
	Matrix * c21 = new Matrix(halfSize,halfSize);
	Matrix * c22 = new Matrix(halfSize,halfSize);
	Matrix * p1 = new Matrix(halfSize,halfSize);
	Matrix * p2 = new Matrix(halfSize,halfSize);
	Matrix * p3 = new Matrix(halfSize,halfSize);
	Matrix * p4 = new Matrix(halfSize,halfSize);
	Matrix * p5 = new Matrix(halfSize,halfSize);
	Matrix * p6 = new Matrix(halfSize,halfSize);
	Matrix * p7 = new Matrix(halfSize,halfSize);
	Matrix * aR = new Matrix(halfSize,halfSize);
	Matrix * bR = new Matrix(halfSize,halfSize);

	int i,j;

	//rozdelnie matic A a B do 4 podmatic
	for (i=0; i<halfSize; i++) {
		for (j=0; j<halfSize; j++) {
			a11->getMatrix()[i][j] = A->getMatrix()[i][j];
			a12->getMatrix()[i][j] = A->getMatrix()[i][j+halfSize];
			a21->getMatrix()[i][j] = A->getMatrix()[i+halfSize][j];
			a22->getMatrix()[i][j] = A->getMatrix()[i+halfSize][j+halfSize];

			b11->getMatrix()[i][j] = B->getMatrix()[i][j];
			b12->getMatrix()[i][j] = B->getMatrix()[i][j+halfSize];
			b21->getMatrix()[i][j] = B->getMatrix()[i+halfSize][j];
			b22->getMatrix()[i][j] = B->getMatrix()[i+halfSize][j+halfSize];
		}
	}

	this->add(a11,a22,aR,halfSize);		//a11+a22
	this->add(b11,b22,bR,halfSize);		//b11+b22
	this->compute(aR,bR,p1,halfSize);	//p1=a11+a22 * b11+b22

	this->add(a21,a22,aR,halfSize);		//a21+a22
	this->compute(aR,b11,p2,halfSize);	//p2=a21+a22 * b11

	this->sub(b12,b22,bR,halfSize);		//b12-b22
	this->compute(a11,bR,p3,halfSize);	//p3=a11 * b12-b22

	this->sub(b21,b11,bR,halfSize);		//b21-b11
	this->compute(a22,bR,p4,halfSize);	//p4=a22 * b21-b11

	this->add(a11,a12,aR,halfSize);		//a11+a12
	this->compute(aR,b22,p5,halfSize);	//p5=a11+a12 * b22

	this->sub(a21,a11,aR,halfSize);		//a21-a11
	this->add(b11,b12,bR,halfSize);		//b11+b12
	this->compute(aR,bR,p6,halfSize);	//p6=a21-a11 * b11+b22

	this->sub(a12,a22,aR,halfSize);		//a12-a22
	this->add(b21,b22,bR,halfSize);		//b21+b22
	this->compute(aR,bR,p7,halfSize);	//p7=a12-a22 * b21+b22

	this->add(p3,p5,c12,halfSize);		//c12=p3+p5
	this->add(p2,p4,c21,halfSize);		//c21=p2+p4

	this->add(p1,p4,aR,halfSize);		//p1+p4
	this->add(aR,p7,bR,halfSize);		//p1+p4+p7
	this->sub(bR,p5,c11,halfSize);		//c11=p1+p4+p7-p5

	this->add(p1,p3,aR,halfSize);		//p1+p3
	this->add(aR,p6,bR,halfSize);		//p1+p3+p6
	this->sub(bR,p2,c22,halfSize);		//c22=p1+p3+p6-p2

	//spojenie do vyslednej matice
	for (i=0; i<halfSize; i++){
		for(j=0; j<halfSize; j++){
			C->getMatrix()[i][j] = c11->getMatrix()[i][j];
			C->getMatrix()[i][j+halfSize] = c12->getMatrix()[i][j];
			C->getMatrix()[i+halfSize][j] = c21->getMatrix()[i][j];
			C->getMatrix()[i+halfSize][j+halfSize] = c22->getMatrix()[i][j];
		}
	}
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

void MatrixList::strassen ()
{
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

		this->compute(m1Resize,m2Resize,mStrassenResize,newSize);

		for (int i=0; i<this->mStrassen->getDimX(); i++)
			for (int j=0; j<this->mStrassen->getDimY(); j++)
				this->mStrassen->getMatrix()[i][j] = mStrassenResize->getMatrix()[i][j];

		delete m1Resize;
		delete m2Resize;
		delete mStrassenResize;
	}
	else {
		this->compute(this->m1,this->m2,this->mStrassen,8);
	}
	this->mStrassen->displayMatrix();

}
