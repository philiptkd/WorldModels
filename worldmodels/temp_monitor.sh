#!/bin/bash

# kills all python processes if the GPU temperature ever exceeds 90C
while [ True ]
do
	if [ $(nvidia-smi | awk -f get_temp.awk) -ge 90 ]
	then
		pkill python
	fi
done
