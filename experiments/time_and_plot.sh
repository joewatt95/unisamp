#!/bin/bash

cd "$(dirname "$0")/time_and_plot" || exit

taskset -c 0,2,4,6,8,10,12,14 poetry run python -m time_and_plot \
  "../../build/unigen_static -e 0.3 --samples {} ../benchmarks/dummy.cnf" \
  -p 8 -r 100 --output-plot ../timing_plot.svg --output-stats ../timing_plot_stats.csv \
  $(seq 10 10 10000)
  # 1.1 1.2 1.3 1.4 1.5 1.52 1.6
  #  "../../build/unigen_static -e 0.3 -r {} --samples 1 ../benchmarks/LoginService.sk_20_34.cnf.gz.no_w.cnf" \
  # "../../build/unigen_static -e 0.3 -r {} --samples 1 ../benchmarks/GuidanceService.sk_4_27.cnf.gz.no_w.cnf" \