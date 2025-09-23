#! /bin/bash

for n in "$@"; do
    i=$n
    k=1
    c=0
    while [ $i -gt 0 ]; do
	a=$(expr $i % 10)
       	if [ $a -ne 0 -a $a -ne 1 ]; then
	    echo "$n n'est pas binaire !"
	    ./readme.sh
	    exit 1
	fi
        c=$(expr $c + $a \* $k)
        k=$(expr $k \* 2)
        i=$(expr $i / 10)
    done
    echo $c
done
