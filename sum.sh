#!/bin/bash

for n in "$@"; do
    tot=0
    c=$(expr $n + 0)
    while [ $c -gt 0 ]; do
	a=$(expr $c % 10)
	c=$(expr $c / 10)
	tot=$(expr $tot + $a)
    done
    echo $tot
done
