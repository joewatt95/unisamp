#!/bin/bash

python time_and_plot.py -r 1 \
  "./build/unigen_static -e 0.3 --samples {} benchmarks/test.cnf" \
  $(seq 0 10 2000)