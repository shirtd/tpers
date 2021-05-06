#!/bin/bash

for set in data/UBLData/*; do
  for test in $set/*; do
    ./main.py --preset --som --plot input pre tpers --save --set $(basename $set) --test $(basename $test)
  done
done
