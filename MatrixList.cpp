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

	int newSize = size / 2;
	Matrix * a11 = new Matrix(newSize,newSize);
	Matrix * a12 = new Matrix(newSize,newSize);
	Matrix * a21 = new Matrix(newSize,newSize);
	Matrix * a22 = new Matrix(newSize,newSize);
	Matrix * b11 = new Matrix(newSize,newSize);
	Matrix * b12 = new Matrix(newSize,newSize);
	Matrix * b21 = new Matrix(newSize,newSize);
	Matrix * b22 = new Matrix(newSize,newSize);
	Matrix * c11 = new Matrix(newSize,newSize);
	Matrix * c12 = new Matrix(newSize,newSize);
	Matrix * c21 = new Matrix(newSize,newSize);
	Matrix * c22 = new Matrix(newSize,newSize);
	Matrix * p1 = new Matrix(newSize,newSize);
	Matrix * p2 = new Matrix(newSize,newSize);
	Matrix * p3 = new Matrix(newSize,newSize);
	Matrix * p4 = new Matrix(newSize,newSize);
	Matrix * p5 = new Matrix(newSize,newSize);
	Matrix * p6 = new Matrix(newSize,newSize);
	Matrix * p7 = new Matrix(newSize,newSize);
	Matrix * aR = new Matrix(newSize,newSize);
	Matrix * bR = new Matrix(newSize,newSize);

	int i,j;

	//rozdelnie matic A a B do 4 podmatic
	for (i=0; i<newSize; i++) {
		for (j=0; j<newSize; j++) {
			a11->getMatrix()[i][j] = A->getMatrix()[i][j];
			a12->getMatrix()[i][j] = A->getMatrix()[i][j+newSize];
			a21->getMatrix()[i][j] = A->getMatrix()[i+newSize][j];
			a22->getMatrix()[i][j] = A->getMatrix()[i+newSize][j+newSize];

			b11->getMatrix()[i][j] = B->getMatrix()[i][j];
			b12->getMatrix()[i][j] = B->getMatrix()[i][j+newSize];
			b21->getMatrix()[i][j] = B->getMatrix()[i+newSize][j];
			b22->getMatrix()[i][j] = B->getMatrix()[i+newSize][j+newSize];
		}
	}

	this->add(a11,a22,aR,newSize);		//a11+a22
	this->add(b11,b22,bR,newSize);		//b11+b22
	this->compute(aR,bR,p1,newSize);	//p1=a11+a22 * b11+b22

	this->add(a21,a22,aR,newSize);		//a21+a22
	this->compute(aR,b11,p2,newSize);	//p2=a21+a22 * b11

	this->sub(b12,b22,bR,newSize);		//b12-b22
	this->compute(a11,bR,p3,newSize);	//p3=a11 * b12-b22

	this->sub(b21,b11,bR,newSize);		//b21-b11
	this->compute(a22,bR,p4,newSize);	//p4=a22 * b21-b11

	this->add(a11,a12,aR,newSize);		//a11+a12
	this->compute(aR,b22,p5,newSize);	//p5=a11+a12 * b22

	this->sub(a21,a11,aR,newSize);		//a21-a11
	this->add(b11,b12,bR,newSize);		//b11+b12
	this->compute(aR,bR,p6,newSize);	//p6=a21-a11 * b11+b22

	this->sub(a12,a22,aR,newSize);		//a12-a22
	this->add(b21,b22,bR,newSize);		//b21+b22
	this->compute(aR,bR,p7,newSize);	//p7=a12-a22 * b21+b22

	this->add(p3,p5,c12,newSize);		//c12=p3+p5
	this->add(p2,p4,c21,newSize);		//c21=p2+p4

	this->add(p1,p4,aR,newSize);		//p1+p4
	this->add(aR,p7,bR,newSize);		//p1+p4+p7
	this->sub(bR,p5,c11,newSize);		//c11=p1+p4+p7-p5

	this->add(p1,p3,aR,newSize);		//p1+p3
	this->add(aR,p6,bR,newSize);		//p1+p3+p6
	this->sub(bR,p2,c22,newSize);		//c22=p1+p3+p6-p2

	//spojenie do vyslednej matice
	for (i=0; i<newSize; i++){
		for(j=0; j<newSize; j++){
			C->getMatrix()[i][j] = c11->getMatrix()[i][j];
			C->getMatrix()[i][j+newSize] = c12->getMatrix()[i][j];
			C->getMatrix()[i+newSize][j] = c21->getMatrix()[i][j];
			C->getMatrix()[i+newSize][j+newSize] = c22->getMatrix()[i][j];
		}
	}
}

void MatrixList::strassen ()
{
	this->mStrassen = new Matrix (m1->getDimX(),m2->getDimY());
	// here goes strassen algorithm
	cout << "fun s nami" << endl;

	this->compute(this->m1,this->m2,this->mStrassen,8);
	this->mStrassen->displayMatrix();

}
