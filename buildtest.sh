#!/bin/bash

for i in `ls -lrtd */|awk -F' ' '{print $9}'`
do
cd $i
pwd
python setup.py bdist_wheel
ls -lrt
cd -
done
