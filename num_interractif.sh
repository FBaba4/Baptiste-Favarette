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


# question 4)b)
#if [ $op = "i" ]; then
#   while true; do
#	echo "saisissez une opération parmi l, m, s, b, d, ou c pour quitter"
#	read choix
#	case "$choix" in
#	    l) for i in $args; do ./len.sh "$i"; done ;;
#	    m) for i in $args; do ./mir.sh "$i"; done ;;
#	    s) for i in $args; do ./sum "$i"; done ;;
#	    b) for i in $args; do ./bin.sh "$i"; done ;;
#	    d) for i in $args; do ./dec.sh "$i"; done ;;
#	    c) echo "Fin du mode interractif"; break ;;
#	    *) echo "Opération non connue"
#	esac
#    done
#fi 




if [ "$op" = "i" ]; then
    mode_interactif=true
    echo "Mode interactif activé."
    echo "Saisissez une opération parmi l, m, s, b, d ; ou c pour quitter."
    read choix
else
    mode_interactif=false
fi

while true; do
    for i in $args; do

	if [ "$choix" = "c" ]; then
	    echo "fin du mode interracif"
	    # break pour 4)c
	    exit 0
	    
	fi
	     case "$choix" in	 
		 l) ./len.sh "$i" ;;
		 m) ./mir.sh "$i" ;;
		 s) ./sum.sh "$i" ;;
		 b) ./bin.sh "$i" ;;
		 d) ./dec.sh "$i" ;;
		 *)
		     echo "Erreur : opération inconnue '$choix'"
		     ./read.sh
		     exit 1 ;;
	     esac
    done
    if [ $mode_interactif = "true" ]; then
	echo "Sélectionner une nouvelle opération parmi l, m, s, b, d, ou c pour quitter le mode intéractif"
	read choix

    else
	break
    fi
done
	     

            
		     
	     
	     
	     
	
	 	  
