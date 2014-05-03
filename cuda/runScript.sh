#/bin/bash

[[ "$HOSTNAME" -eq "node-051" ]] && GPU="gtx470"
[[ "$HOSTNAME" -eq "node-052" ]] && GPU="gtx480"
[[ "$HOSTNAME" -eq "node-053" ]] && GPU="gtx590"
[[ "$HOSTNAME" -eq "node-054" ]] && GPU="teslak40"


for I in 1 2 3
do
	for INPUT in 256 512 1024 2048 4096 8192
	do
		./cuda -f /mnt/data/klepamar/inputs/input_${INPUT} > /mnt/data/klepamar/${GPU}_result_${INPUT}_${I}
		echo "Processed $INPUT, run no: $I"
	done
done
