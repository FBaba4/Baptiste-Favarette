#! /bin/bash

chmod u+x readme.sh

if [ $# -lt 2 ]  ; then
    ./readme.sh
    

fi 


op="$1"
shift

case "$op" in
    len) op="l" ;;
    mir) op="m" ;;
    sum) op="s" ;;
    bin) op="b" ;;
    dec) op="d" ;;
    int) op="i" ;;
esac



args="$@"

for arg in $args; do
    isnum="$(expr 0 + "$arg" 2>/dev/null)"
    if [ -z "$isnum" ]; then
        echo "Erreur : '$arg' n'est pas un entier"
        ./readme.sh
	exit 1
    fi
done


echo "Opération : $op"
echo "Arguments : $args" 


for arg in $args; do
    if [ $arg -lt 0 ]; then
	echo "Les arguments doivent être positifs !"
	./readme.sh
	exit 1

    fi
done


for i in $args; do
    case "$op" in
	l) ./len.sh "$i" ;;
	m) ./mir.sh "$i" ;;
	s) ./sum.sh "$i" ;;
	b) ./bin.sh "$i" ;;
	d) ./dec.sh "$i" ;;
	i) ./int.sh "$i" ;;
	*) echo "Erreur: opération inconnue au bataillon '$op'"
	   ./readme.sh
	   exit 1
    esac
done
    
