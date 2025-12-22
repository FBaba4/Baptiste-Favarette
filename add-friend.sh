#!/bin/bash

if [ $# -ne 2 ]; then
    echo "erreur d'arguments"
    exit 1
fi

id=$1
friend=$2

./P.sh $id.lock
if !  [ -d $id ];then
    echo "nok user "$id" does not exist"
    ./V.sh $id.lock
    exit 1
fi


if ! [ -d $friend ]; then
    echo "nok user "$friend" does not exist"
    ./V.sh $id.lock
    exit 1
fi


if  grep -q  "^$friend$" "$id"/friends; then
    echo "nok user $friend is already a friend of $id"
    ./V.sh $id.lock
    exit 1
fi


echo $friend >> $id/friends
./V.sh $id.lock
echo ok
exit 0






    
    
