#! /bin/bash

for i in "$@"; do
    n=$i
    n=$(expr $n + 0)
    c=0

    if [ "$n" -eq 0 ]; then
        c=1
    else
        while [ "$n" -gt 0 ]; do
            n=$((n / 10))  
            c=$((c + 1))
        done
    fi

    echo "$c"
done

