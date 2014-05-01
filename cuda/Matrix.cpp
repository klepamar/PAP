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
	this->matrix = new int*[this->dimX];
    for (int i=0; i<this->dimX; i++) 
        matrix[i] = new int[this->dimY];
        
    for (int i=0; i<this->dimX; i++) // clear the matrix before any calculation
		for (int j=0; j<this->dimY; j++)
			matrix[i][j] = 0;
}

Matrix::~Matrix ()
{
	for (int i=0; i<this->dimX; i++) 
	{
		delete[] matrix[i]; // delete array of int
        matrix[i] = NULL;
    }
    delete[] matrix; // delete array of int*
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

int** Matrix::getMatrix() const
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
			matrix[i][j] = inputElement;
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
				cout << &(matrix[i][j]) << endl;
			else
				cout << &(matrix[i][j]) << " ";
		}
	}
	cout << "</HOW MATRIX IS INTERNALLY SAVED>" << endl;
}

void Matrix::showMatrix() const {
	cout << "dimensions (x,y): " << this->dimX << "," << this->dimY << endl;
	for (int i=0; i<this->dimX; i++) {
		for (int j=0; j<this->dimY; j++)
			cout << setw(6) << this->matrix[i][j];
		cout << endl;
	}
}

void Matrix::displayMatrix() const
{
	cout << "<MATRIX>" << endl;
	cout << "dimensions (x,y): " << this->dimX << "," << this->dimY << endl;
	
	cout << "┌"; //top left border
    for (int i=0; i<dimY-1; i++) // top border 
    {
		cout << "──────┬";
    }
	cout << "──────┐" << endl; // top right border
	
	for (int i=0; i<dimX; i++)
	{
		cout << "|"; // left border
		for (int j=0; j<dimY; j++)
		{
			if ((j+1) == dimY) cout << setw(5) << matrix[i][j] << " |" << endl;
			else cout << setw(5) << matrix[i][j] << " |";
		}
		// borders in between two rows
		if (i != dimX - 1) 
		{
			cout << "├";
            for (int i=0; i<dimY-1; i++) 
            {
				cout << "──────┼";
            }
			cout << "──────┤" << endl;
        }
	}
	cout << "└"; // bottom left border
    for (int i=0; i<dimY-1; i++) // bottom border 
    {
		cout << "──────┴";
    }
    cout << "──────┘" << endl; // bottom right border
	cout << "</MATRIX>" << endl;
}
