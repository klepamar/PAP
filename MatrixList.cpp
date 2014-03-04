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

void MatrixList::strassen ()
{
	// here goes strassen algorithm
}
