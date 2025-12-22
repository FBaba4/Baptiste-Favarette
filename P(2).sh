#! /bin/bash

if [ -z "$1" ]; then
		echo "Usage $0 mutex-name" >&2
		exit 1
else
		# the POSIX 'link' system call is supposed to be atomic on a local file system
		# an nfs server is also supposed to execute it atomically (at least since the v2)

                while ! ln "$0" "$1" 2>/dev/null; do

				sleep 0
		done


		# On some filesystems (eg NTFS), ln may create a copy of the file instead of a hard link.
		# Thus, if another process calls P.sh, ln will overwrite the file instead of failing
		# We need to check if the created file is a hard link
		inode_src=$(ls -i "$0" |cut -f 1 -d" ")
		inode_dest=$(ls -i "$1" |cut -f 1 -d" ")

		if [ "$inode_src" != "$inode_dest" ]; then
		    echo "Error: P.sh does not work on your filesystem!" >&2
		    exit 1
		fi
		exit 0
fi
