#!/bin/bash
for i in $(seq 0 11)
do
  ./puzzle $i > out/$i.log &
done
