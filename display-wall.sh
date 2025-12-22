#!/bin/bash

id=$1

if [ $# -ne 1 ]; then
    echo "erreur d'arguments"
    exit 1
fi


#test si le fichier existe bien


if ! [ -d $id ]; then
    echo "nok user "$id" does not exist"
    exit 1
fi

#affichage du mur
./P.sh $id.lock
echo "start-of-file" # \n $(cat $id/wall) \n "end-of-file
cat $id/wall
echo "end-of-file"
./V.sh $id.lock
