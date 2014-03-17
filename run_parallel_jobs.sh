#!/bin/bash

# vystup - spustenie paralelnych jobov vygenerovanych skriptom create_parallel_jobs.sh

# make sure correct method is being invoked in index.cpp - change index.cpp & run make afterwards

for INPUT_SIZE in 1024 2048 4096
do
	for THREAD_NO in 1 2 4 6 8 12
	do
		/opt/bin/qrun.sh 24c 1 1slots_per_host parallel_job_${INPUT_SIZE}_${THREAD_NO}.sh
	done
done

echo "Output of respective programs will be stored in /mnt/data/$USERNAME as defined in ./create_parallel_jobs.sh script."
