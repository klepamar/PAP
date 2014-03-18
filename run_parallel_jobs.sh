#!/bin/bash

# vystup - spustenie paralelnych jobov vygenerovanych skriptom create_parallel_jobs.sh

for I in 1 2 3 # repeat 3 times for more precise results
do
	for PROGRAM in classic optimised # add strassen64 when parallel implementation is ready
	do
		for INPUT_SIZE in 1024 2048 4096 # for all available inputs
		do
			for THREAD_NO in 1 2 4 6 8 12 24 # for given number of threads
			do
				/opt/bin/qrun.sh 24c 1 1slots_per_host parallel_job_${PROGRAM}_${INPUT_SIZE}_${THREAD_NO}.sh
			done
		done
	done
done

echo "Output of respective programs will be stored in /mnt/data/$USERNAME as defined in ./create_parallel_jobs.sh script."
