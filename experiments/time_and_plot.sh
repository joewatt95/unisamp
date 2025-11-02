#!/bin/bash

taskset -c 0,2,4,6,8,10,12,14 \
  poetry --directory "$(dirname "$0")" --project ~/dev/time_and_plot \
    run python -m time_and_plot \
      "../build/unigen_static -e 0.3 --samples 100 benchmarks/{}.cnf" \
      -p 8 -r 20 --output-plot timing_plot.svg --output-stats timing_plot_stats.csv \
      "GuidanceService.sk_4_27.cnf.gz.no_w" "LoginService.sk_20_34.cnf.gz.no_w"
      # $(seq 10 10 5000)