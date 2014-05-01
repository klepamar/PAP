#include <iostream>
#include <string>
#include <stdlib.h>
#include <iomanip>

#include "Matrix.h"

using namespace std;

extern char* result;
extern ofstream resultStream;

Matrix::Matrix (int x, int y)
{
	this->dimX = x;
	this->dimY = y;
	this->matrix = new int[x*y];

	for (int j=0; j<(x*y); j++)
		matrix[j] = 0;
}

Matrix::~Matrix ()
{
    delete[] matrix; // delete array of int
    matrix = NULL;
}

int Matrix::getDimX () const
{
	return this->dimX;
}

int Matrix::getDimY () const
{
	return this->dimY;
}

int* Matrix::getMatrix() const
{
	return this->matrix;
}

void Matrix::fillMatrix(istream &in)
{
	int inputElement;
	string dummy;
	
	for (int i=0; i<dimX; i++)
	{
		for (int j=0; j<dimY; j++)
		{
			in >> inputElement;
			matrix[i * this->getDimX() + j] = inputElement;
		}
		getline(in,dummy); // get rid of new line character
	}
}

void Matrix::displayAddresses() const
{
	cout << "<HOW MATRIX IS INTERNALLY SAVED>" << endl;
	for (int i=0; i<dimX; i++)
	{
		for (int j=0; j<dimY; j++)
		{
			if ((j+1) == dimY)
				cout << &(matrix[i * this->getDimX() + j]) << endl;
			else
				cout << &(matrix[i * this->getDimX() + j]) << " ";
		}
	}
	cout << "</HOW MATRIX IS INTERNALLY SAVED>" << endl;
}

void Matrix::showMatrix() const {
	cout << "dimensions (x,y): " << this->dimX << "," << this->dimY << endl;
	for (int i=0; i<this->dimX; i++) {
		for (int j=0; j<this->dimY; j++)
			cout << setw(6) << this->matrix[i * this->getDimX() + j];
		cout << endl;
	}
}
