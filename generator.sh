#!/bin/bash

# dimensions: m x n, n x p
# maximalna dimenzia matic = MAXSIZE
# znamienko = SIGN
# maximalna hodnota prvku v matici = RANGE

MAXSIZE=20
RANGE=10

# in input parameters are not entered, generate a matrix with random size
# to prevent from small/zero values, add a small constant = 3
if [[ "$#" -ne 3 ]]
then
	echo "Random matrix dimensions will be used. Otherwise, ./generator m n p"
	M=$((RANDOM % $MAXSIZE + 3))
	N=$((RANDOM % $MAXSIZE + 3))
	P=$((RANDOM % $MAXSIZE + 3))
else
	M=$1
	N=$2
	P=$3
fi

# generuj maticu A

# zobraz pocet riadkov & stlpcov
echo "$M $N"
for ((i=0; i<$M; i++)) # for riadky
do
	for ((j=0; j<$N; j++)) # for stlpce
	do
		NUMBER=$((RANDOM % RANGE))
		SIGN=$((RANDOM % 2))
		if [[ $((j+1)) -eq $N ]]
		then
			echo -ne "$((NUMBER-2*SIGN*NUMBER))\n"
		else
			echo -ne "$((NUMBER-2*SIGN*NUMBER)) "
		fi
	done
done

# generuj maticu B

# zobraz pocet riadkov & stlpcov
echo "$N $P"
for ((i=0; i<$N; i++)) # for riadky
do
	for ((j=0; j<$P; j++)) # for stlpce
	do
		NUMBER=$((RANDOM % RANGE))
		SIGN=$((RANDOM % 2))
		if [[ $((j+1)) -eq $P ]]
		then
			echo -ne "$((NUMBER-2*SIGN*NUMBER))\n"
		else
			echo -ne "$((NUMBER-2*SIGN*NUMBER)) "
		fi
	done
done
