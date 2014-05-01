CC=nvcc
#CFLAGS=-fopenmp -O3

all: index

index: index.o Matrix.o
	$(CC) $(CFLAGS) index.o Matrix.o -o a.out

index.o: index.cu
	$(CC) $(CFLAGS) -c index.cu

Matrix.o: Matrix.cpp
	$(CC) $(CFLAGS) -c Matrix.cpp

clean:
	rm -rf *.o
