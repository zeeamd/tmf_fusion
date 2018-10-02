#!/bin/bash

u=`cat /var/lib/jenkins/artifactory/.AU`
p=`cat /var/lib/jenkins/artifactory/.P`

echo $1
echo $2

#recursive python wheel creation
for d in `ls -lrtd */|awk -F' ' '{print $9}'`
do
cd $d
echo $1 $2
curl -u $1:$2 -T dist/extracts-_python_version_-py2-none-any.whl "http://18.188.209.12:8081/artifactory/generic-local/"
cd -
done
