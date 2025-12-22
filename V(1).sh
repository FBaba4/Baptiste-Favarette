#! /bin/bash

if [ -z "$1" ]; then
		echo "Usage $0 mutex-name" >&2
		exit 1
else
		rm -f "$1"
		exit 0
fi
