#!/bin/bash

if [ $# -lt 3 ]; then
    echo "erreur d'arguments, il en faut au minimun 3 !"
    exit 1
fi

sender=$1
receiver=$2


if ! [ -d $sender ]; then
    echo "nok user "$sender" does not exist"
    exit 1
fi


./P.sh $receiver.lock

if ! [ -d $receiver ]; then
    echo "nok user "$receiver" does not exist"
    ./V.sh $receiver.lock
    exit 1
    
fi


if ! grep -q "^$receiver$" "$sender/friends"; then
    echo "nok user "$sender" is not a friend of '$receiver'"
    ./V.sh $receiver.lock
    exit 1
fi

shift 2
echo "$sender: $@" >> $receiver/wall
./V.sh $receiver.lock
echo ok
exit 0


    
