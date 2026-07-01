#! /bin/bash

if [ -z "$2" ]; then
    echo Bonjour marcheur blanc !
else
    while [ $# -gt 0 ] ; do
	echo "Bonjour $1"
	shift
	shift

    done
    
fi

       

