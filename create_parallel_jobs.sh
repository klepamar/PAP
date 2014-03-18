#!/bin/bash

# vystup - subory parallel_jobs.sh pre jednotlive vstupy & pocet vytvaranych vlakien

if [[ "$#" -ne 1 ]]
then
	echo "Enter your username as the only argument."
	exit 1
else
	USERNAME=$1
fi

for PROGRAM in classic optimised strassen64
do
	for INPUT_SIZE in 1024 2048 4096
	do
		for THREAD_NO in 1 2 4 6 8 12 24
		do
			cp /home/mpi/parallel_job.sh ./parallel_job_${PROGRAM}_${INPUT_SIZE}_${THREAD_NO}.sh
			sed -i 's/INFINIBAND="true"/INFINIBAND="false"/g' ./parallel_job_${PROGRAM}_${INPUT_SIZE}_${THREAD_NO}.sh 
			sed -i "s/#$ -e ./#$ -e \/mnt\/data\/$USERNAME/g" ./parallel_job_${PROGRAM}_${INPUT_SIZE}_${THREAD_NO}.sh
			sed -i "s/#$ -o ./#$ -o \/mnt\/data\/$USERNAME/g" ./parallel_job_${PROGRAM}_${INPUT_SIZE}_${THREAD_NO}.sh
			sed -i "s/MY_PARALLEL_PROGRAM=\(.*\)/MY_PARALLEL_PROGRAM=\".\/${PROGRAM} -f input_${INPUT_SIZE} -v -n ${THREAD_NO}\"/g" \
			./parallel_job_${PROGRAM}_${INPUT_SIZE}_${THREAD_NO}.sh
		done
	done
done
