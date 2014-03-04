#ifndef MATRIXLIST_H
#define	MATRIXLIST_H

#include "Matrix.h"

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
public:
	MatrixList (Matrix* m1, Matrix* m2);
	
	virtual ~MatrixList();

	bool isCorrectSize() const;
	void classic();
	void classicOptimised();
	void strassen();
};

#endif	/* MATRIXLIST_H */
