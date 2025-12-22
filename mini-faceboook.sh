#!/bin/bash

req=$1
id=$2
shift 2
args=$@


while true; do
    read req id args


    case $req in
	"create")
	    ./create.sh $id
	    ;;
	"add")
	    ./add-friend.sh $id $args
	    ;;
	"display")
	    ./display-wall.sh $id
	    ;;
	"post")
	    ./post-message.sh $id $args
	    ;;
	*)
	    echo "nok bad request"
	    ;;
	
    esac

done
