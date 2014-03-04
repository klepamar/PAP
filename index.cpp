#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include "Matrix.h"
#include "MatrixList.h"

using namespace std;

/* DEFINE "GLOBAL" VARIABLES HERE */

char* filename; // where input matrixes are stored
bool verbose = false; // produce debugging messages with -v flag
int block = 100; // block size used for loop tiling in classical approach

/* END OF GLOBAL VARIABLES DEFINITION */

/* read source data from input file & initialise matrixes */
void readInputFile (Matrix* &m1, Matrix* &m2, MatrixList* &ml)
{
	int x,y; // dimensions
	string dummy;
	ifstream in;
	
	in.open(filename); // open file
	if (!in.is_open()) throw "could not open file.";

	in >> x >> y; // read the dimensions of m1
	getline(in,dummy); // get rid of new line character
	m1 = new Matrix(x,y);
	m1->fillMatrix(in); // fill in m1
	
	in >> x >> y; // the same for m2
	getline(in,dummy);
	m2 = new Matrix(x,y);
	m2->fillMatrix(in);
	
	in.close(); // end up processing file
	
	ml = new MatrixList(m1,m2); // create a new object - MatrixList, which includes both matrixes
}

/* display help when invalid option has been used */
void displayHelp ()
{
	cout << "Usage:" << endl <<
			"\t-f FILE\tinput file with matrixes" << endl <<
			"\t-v\tverbose mode (optional)" << endl;
}

/* determine input file and possibly other arguments */
void processArguments (int argc, char** argv)
{
	if (argc == 1) // no arguments provided
	{
		displayHelp();
		throw "no arguments provided.";
	}
	for (int i=1; i<argc; i++) // first argument argv[0] is executable
	{
		if (strcmp(argv[i],"-f") == 0) // -f...input file
		{
			if ((i+1) == argc) throw "no input file provided.";
			filename = argv[i+1];
			i++; // next "i" is already resolved
		}
		else if (strcmp(argv[i],"-v") == 0) // -v...verbose mode
		{
			verbose = true;
		}
		else
		{
			displayHelp();
			throw "unknown arguments provided.";
		}
	}
}

void verifyMatrixSize(MatrixList* &ml)
{
	// verify that multiplication can be performed
	if (!ml->isCorrectSize())
		throw "Matrixes must of size m x n and n x p respectively";
}

int main(int argc, char** argv)
{
	//double start=omp_get_wtime();
	//double end;
	
	Matrix* m1=NULL;
	Matrix* m2=NULL;
	MatrixList *ml=NULL;
	try
	{
		processArguments(argc,argv);
		readInputFile(m1,m2,ml);
		verifyMatrixSize(ml);
	}
	catch (const char* exception)
	{
		cout << "Exception: " << exception << endl;
		delete m1; // matrixes created in readInputFile method
		delete m2;
		delete ml; // matrix list containing both matrixes
		exit (EXIT_FAILURE);
	}
	
	//m1->displayAddresses();
	ml->classic();
	ml->classicOptimised();
	
	delete m1; // matrixes created in readInputFile method
	delete m2;
	delete ml; // matrix list containing both matrixes - in readInputFile
	
	//end=omp_get_wtime();
	//if (verbose) cout << "Calculation took " << (end-start) << " seconds." << endl;
	
	return (EXIT_SUCCESS);
}
