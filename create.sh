#! /bin/bash

if [ $# -eq 0 ]; then
    echo "nok please provide an identifier"
    exit 1
fi


id=$1
./P.sh $id.lock    
if [  -d $id ]; then
    echo "nok user "$id" already exists"

else
    mkdir $id
    touch $id/wall
    touch $id/friends
    echo ok
    
fi
./V.sh $id.lock

    
    
    
