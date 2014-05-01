#ifndef MATRIX_H
#define	MATRIX_H

#include <iostream>

class Matrix {
private:
	int dimX, dimY;
	int* matrix;
public:
	Matrix(int x, int y);
	
	virtual ~Matrix();
	
	void fillMatrix(std::istream &in);
	void showMatrix() const;
	void displayAddresses() const;
	int getDimX() const;
	int getDimY() const;
	int* getMatrix() const;
};

#endif	/* MATRIX_H */
