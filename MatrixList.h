#ifndef MATRIXLIST_H
#define	MATRIXLIST_H

#include "Matrix.h"

class MatrixList
{
private:
	Matrix *m1;
	Matrix *m2;
	Matrix *mClassic;
	Matrix *mStrassen;
public:
	MatrixList (Matrix* m1, Matrix* m2);
	
	virtual ~MatrixList();

	bool isCorrectSize() const;
	void classic();
	void strassen();
};

#endif	/* MATRIXLIST_H */
