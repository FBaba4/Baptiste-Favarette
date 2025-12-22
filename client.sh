#!/bin/bash

id=$1
mkfifo $id.pipe
exec 3<> $id.pipe

if [ $# -eq  0 ]; then
    echo "erreur d'arguments"
    exit 1
fi


while true; do
    read req args
    if ! [ -z $req ]; then
	echo $req $id $args
    fi
done
