#! /bin/bash
for i in "$@"; do
    n=$i
    c=$(expr $n + 0)
    b=0
    while [ "$c" -gt 0 ]; do
	a=$(expr $c % 10)
	c=$(expr $c / 10)
	b=$(expr $b \* 10 + $a)
    done
    echo $b
done

