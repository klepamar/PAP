CC=g++
CFLAGS=

all: index

index: index.o Matrix.o MatrixList.o 
	$(CC) $(CFLAGS) index.o Matrix.o MatrixList.o -o a.out

index.o: index.cpp
	$(CC) $(CFLAGS) -c index.cpp

Matrix.o: Matrix.cpp
	$(CC) $(CFLAGS) -c Matrix.cpp

MatrixList.o: MatrixList.cpp
	$(CC) $(CFLAGS) -c MatrixList.cpp

clean:
	rm -rf *.o

