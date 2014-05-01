#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include "Matrix.h"

using namespace std;

/* DEFINE "GLOBAL" VARIABLES HERE */

char* filename; // where input matrixes are stored
bool verbose = false; // produce debugging messages with -v flag
int block = 100; // block size used for loop tiling in classical approach
int cpuNumber; // number of processesors active during OpenMP execution

/* END OF GLOBAL VARIABLES DEFINITION */

/* read source data from input file & initialise matrixes */
void readInputFile (Matrix* &m1, Matrix* &m2)
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
}

/* display help when invalid option has been used */
void displayHelp ()
{
	cout << "Usage:" << endl <<
			"\t-f FILE\tinput file with matrixes" << endl <<
			"\t-n NO_OF_CPUS\t number of active CPUs" << endl <<
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
		else if (strcmp(argv[i],"-n") == 0) // -n...number of active CPUs
		{
			cpuNumber = atoi(argv[i+1]);
			i++;
		}
		else
		{
			displayHelp();
			throw "unknown arguments provided.";
		}
	}
}

void multiply (int *matrixA, int *matrixB, int *matrixC, int matrixSize) {
	cout << "hello world" << endl;
}

int main(int argc, char** argv)
{
	
	Matrix* m1=NULL;
	Matrix* m2=NULL;
	try
	{
		processArguments(argc,argv);
		readInputFile(m1,m2);
	}
	catch (const char* exception)
	{
		cout << "Exception: " << exception << endl;
		delete m1; // matrixes created in readInputFile method
		delete m2;
		exit (EXIT_FAILURE);
	}
	
	Matrix * m3 = new Matrix(m1->getDimX(),m2->getDimY());
	
	multiply(*m1->getMatrix(),*m2->getMatrix(),*m3->getMatrix(),m1->getDimX());
	
	delete m1; // matrixes created in readInputFile method
	delete m2;
	delete m3; 
	
	
	return (EXIT_SUCCESS);
}
