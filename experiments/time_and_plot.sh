#!/bin/bash

taskset -c 0,2,4,6,8,10,12,14 \
  poetry --directory "$(dirname "$0")" --project ~/dev/time_and_plot \
    run python -m time_and_plot \
      "../build/unisamp_static -e 0.3 -r 1.6 --samples {} benchmarks/dummy.cnf" \
      -p 8 -r 20 --output-plot timing_plot.svg --output-stats timing_plot_stats.csv \
      $(seq 50 50 5000)