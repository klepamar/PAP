CC=nvcc
CFLAGS=-O3

all: index

index: index.o Matrix.o
	$(CC) $(CFLAGS) index.o Matrix.o -o cuda

index.o: index.cu
	$(CC) $(CFLAGS) -c index.cu

Matrix.o: Matrix.cpp
	$(CC) $(CFLAGS) -c Matrix.cpp

clean:
	rm -rf *.o
