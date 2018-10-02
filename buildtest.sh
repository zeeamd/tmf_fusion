#!/bin/bash

#recursive python wheel creation
for d in `ls -lrtd */|awk -F' ' '{print $9}'`
do
cd $d
python setup.py bdist_wheel
cd -
done
