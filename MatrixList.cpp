#include "MatrixList.h"

MatrixList::MatrixList (Matrix* m1, Matrix* m2)
{
	this->m1=m1; // or with constructor?
	this->m2=m2;
	this->mClassic = this->mStrassen = NULL;
}

MatrixList::~MatrixList()
{
	delete mClassic;
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

	for (int i=0; i<m1->getDimX(); i++) // m1-riadok
		for (int j=0; j<m2->getDimY(); j++) // m2-stlpec
			for (int k=0; k<m1->getDimY(); k++) // m1-stlpec == m2-riadok
				this->mClassic->getMatrix()[i][j] += this->m1->getMatrix()[i][k] * this->m2->getMatrix()[k][j];
				
	mClassic->displayMatrix();
}

void MatrixList::strassen ()
{
	// here goes strassen algorithm
	mClassic = new Matrix (m2->getDimY(),m1->getDimX());
}
