#ifndef MATRIXLIST_H
#define	MATRIXLIST_H

#include "Matrix.h"
#include "math.h"

class MatrixList
{
private:
	Matrix *m1;
	Matrix *m2;
	Matrix *mClassic;
	Matrix *mClassicOptimised;
	Matrix *mStrassen;
	double start_timer, end_timer;
	double classic_length, optimised_length;

	void add(Matrix *A, int A_off_X, int A_off_Y, Matrix *B, int B_off_X, int B_off_Y, Matrix *C, int C_off_X, int C_off_Y, int size);
	void sub(Matrix *A, int A_off_X, int A_off_Y, Matrix *B, int B_off_X, int B_off_Y, Matrix *C, int C_off_X, int C_off_Y, int size);
	void mul(Matrix *A, int A_off_X, int A_off_Y, Matrix *B, int B_off_X, int B_off_Y, Matrix *C, int C_off_X, int C_off_Y, int size);

	void compute(Matrix *A, int A_off_X, int A_off_Y, Matrix *B, int B_off_X, int B_off_Y, Matrix *C, int C_off_X, int C_off_Y, int size);
	int nextPowerOf2(int number) const;
	int maxDim() const;
public:
	MatrixList (Matrix* m1, Matrix* m2);
	
	virtual ~MatrixList();

	bool isCorrectSize() const;
	void classic();
	void classicOptimised();
	void strassen();
	void cuda();
};

#endif	/* MATRIXLIST_H */
