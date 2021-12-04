#!/bin/bash

declare -A pids=()

for i in $(seq 0 12)
do
  ./puzzle $i 13 > out/$i-solutions.log &
  pids[$i]=$!
done

for i in $(seq 0 12)
do
  wait ${pids[$i]}
done
