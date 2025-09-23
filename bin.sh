#! /bin/bash



for n in "$@"; do
    a=$(expr $n + 0)
    b=0
    k=1
    if [ $a -eq 0 ]; then
	echo 0
    else
	
	while [ $a -gt 0 ]; do
	    c=$(expr $a % 2)
	    a=$(expr $a / 2)
	    b=$(expr $b + $c \* $k)
	    k=$(expr $k \* 10)
	done
	echo $b
    fi   
done 
