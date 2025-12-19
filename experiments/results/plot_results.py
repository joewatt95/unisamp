#!/usr/bin/env python3
"""
plot_results.py (Unified: Time + Samples)

Features:
- **Time Analysis**: PAR-2 scores, Time CDF, Time Box Plots.
- **Sample Analysis**: Solution Count Scatter (with Time coloring), Throughput Curves.
- **Unified**: Generates all plots in one run.

python results_new.csv --baseline results_old.csv -o out
"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ================= CONFIGURATION =================
TIMEOUT = 3600      
PAR_PENALTY = 2     
BASELINE_NAME = "Baseline (Old)"
OUTPUT_FORMAT = "svg"
# =================================================

def load_and_unify_data(main_csv, param_col, baseline_csv=None):
    df_main = pd.read_csv(main_csv)
    
    # 1. Standardize 'Solver_Version' Label
    if param_col and param_col in df_main.columns:
        df_main['Solver_Version'] = df_main[param_col].apply(lambda x: f"{param_col}={x}")
    elif {'r', 'e', 'samples'}.intersection(df_main.columns):
        cols = [c for c in ['r', 'e', 'samples'] if c in df_main.columns]
        df_main['Solver_Version'] = df_main[cols].apply(
            lambda x: "_".join(f"{k}={v}" for k, v in x.items()), axis=1
        )
    else:
        df_main['Solver_Version'] = "New_Version"

    if baseline_csv:
        df_base = pd.read_csv(baseline_csv)
        df_base['Solver_Version'] = BASELINE_NAME
        df_final = pd.concat([df_main, df_base], ignore_index=True)
    else:
        df_final = df_main

    # 2. Setup Time Metric (CPU Priority)
    if 'CPU_Time' in df_final.columns:
        time_metric = 'CPU_Time'
    elif 'Wall_Time' in df_final.columns:
        time_metric = 'Wall_Time'
    else:
        time_metric = 'Real_Time'

    # Clean numeric time
    df_final[time_metric] = pd.to_numeric(df_final[time_metric], errors='coerce')

    # 3. Solved Status Logic
    df_final['Solved'] = df_final['Outcome'].astype(str).apply(
        lambda x: "success" in x.lower() or "unsat" in x.lower()
    )

    # 4. PAR-2 Calculation
    df_final['Time_PAR2'] = df_final[time_metric]
    df_final.loc[~df_final['Solved'], 'Time_PAR2'] = TIMEOUT * PAR_PENALTY
    
    df_final['Time_Capped'] = df_final[time_metric]
    df_final.loc[~df_final['Solved'], 'Time_Capped'] = TIMEOUT
    df_final.loc[df_final['Time_Capped'] > TIMEOUT, 'Time_Capped'] = TIMEOUT

    # 5. Parse Solution Counts (For Partial Progress Plots)
    # Check multiple potential column names
    sol_col = None
    for c in ['Solutions_Found', 'found', 'samples']:
        if c in df_final.columns:
            sol_col = c
            break
            
    if sol_col:
        df_final['Samples_Found'] = pd.to_numeric(df_final[sol_col], errors='coerce').fillna(0).astype(int)
    else:
        df_final['Samples_Found'] = 0

    return df_final

def print_detailed_stats(df):
    print("\n" + "="*80)
    print(f"{'VERSION':<30} | {'PAR-2':<8} | {'SOLVED':<8} | {'AVG SAMPLES'}")
    print("="*80)

    for version, group in df.groupby('Solver_Version'):
        total = len(group)
        solved = group['Solved'].sum()
        par2 = group['Time_PAR2'].mean()
        avg_samples = group['Samples_Found'].mean()
        
        # Breakdown Failures
        timeouts = group[group['Outcome'].str.contains('Timeout', na=False)]
        t_total = len(timeouts)
        
        print(f"{version:<30} | {par2:<8.1f} | {solved}/{total} | {avg_samples:.1f}")
    print("="*80 + "\n")

# ================= PLOTTING FUNCTIONS =================

def plot_cdf_time(df, output_dir):
    """ Standard Time-based CDF """
    plt.figure(figsize=(10, 7))
    versions = sorted(df['Solver_Version'].unique())
    palette = sns.color_palette("tab10", n_colors=len(versions))

    for i, version in enumerate(versions):
        group = df[df['Solver_Version'] == version]
        solved_times = sorted(group[group['Solved']]['Time_Capped'].tolist())
        
        if not solved_times: continue

        y_axis = range(1, len(solved_times) + 1)
        
        if version == BASELINE_NAME:
            plt.plot(solved_times, y_axis, color='black', linewidth=2.5, 
                     linestyle='--', label=version, zorder=10)
        else:
            plt.plot(solved_times, y_axis, color=palette[i], 
                     linewidth=2, alpha=0.9, label=version)

    plt.xscale('log')
    plt.xlabel("CPU Time (s) [Log Scale]", fontsize=12)
    plt.ylabel("Solved Instances (Cumulative)", fontsize=12)
    plt.title("Solved Instances over Time (CDF)", fontsize=14)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"cdf_time.{OUTPUT_FORMAT}")
    print(f"Generated cdf_time.{OUTPUT_FORMAT}")

def plot_scatter_samples(df, output_dir):
    """
    Scatter: Baseline Samples vs Best New Samples.
    COLOR: Encodes Time Speedup.
    """
    if BASELINE_NAME not in df['Solver_Version'].values: return

    # Identify Best New Version (Lowest PAR-2)
    ranking = df.groupby('Solver_Version')['Time_PAR2'].mean().sort_values()
    best_new = None
    for version in ranking.index:
        if version != BASELINE_NAME:
            best_new = version
            break
            
    if not best_new: return
        
    print(f"Sample Scatter: Comparing '{BASELINE_NAME}' vs '{best_new}'")

    # Pivot Data
    # We need both Samples and Time for coloring
    pivot_samp = df.pivot(index='Input_File', columns='Solver_Version', values='Samples_Found')
    pivot_time = df.pivot(index='Input_File', columns='Solver_Version', values='Time_Capped')
    
    data_samp = pivot_samp[[BASELINE_NAME, best_new]].dropna()
    data_time = pivot_time[[BASELINE_NAME, best_new]].dropna()
    
    # Calculate Speedup for Coloring (Base Time / New Time)
    # > 1 means New is Faster (Green), < 1 means Slower (Red)
    # We use Log scale for color map
    speedup = data_time[BASELINE_NAME] / data_time[best_new]

    plt.figure(figsize=(8, 7))
    
    # Jitter to show density at (0,0) and (100,100)
    jitter_x = data_samp[BASELINE_NAME] + (np.random.rand(len(data_samp)) - 0.5) * 2
    jitter_y = data_samp[best_new] + (np.random.rand(len(data_samp)) - 0.5) * 2

    sc = plt.scatter(jitter_x, jitter_y, c=np.log10(speedup), cmap='RdYlGn', 
                     vmin=-1, vmax=1, alpha=0.7, s=20, edgecolors='grey', linewidth=0.3)

    # Diagonal
    max_val = max(data_samp.max().max(), 100)
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label="Equal Samples")

    cbar = plt.colorbar(sc)
    cbar.set_label("Speedup (Log Scale)\nGreen=New Faster, Red=New Slower", rotation=270, labelpad=20)

    plt.xlabel(f"Samples Found: {BASELINE_NAME}", fontsize=11)
    plt.ylabel(f"Samples Found: {best_new}", fontsize=11)
    plt.title(f"Head-to-Head: Samples Found\n(Color indicates Timing Speedup)", fontsize=13)
    
    plt.axis('square')
    plt.xlim(-5, max_val + 5)
    plt.ylim(-5, max_val + 5)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(output_dir / f"scatter_samples_colored.{OUTPUT_FORMAT}")
    print(f"Generated scatter_samples_colored.{OUTPUT_FORMAT}")

def plot_throughput_curve(df, output_dir):
    """
    Plots 'Sorted Sample Yield'.
    X-axis: Top N Benchmarks
    Y-axis: Samples Found (Sorted Descending)
    Shows which solver maintains high sample yield on more instances.
    """
    plt.figure(figsize=(10, 7))
    versions = sorted(df['Solver_Version'].unique())
    palette = sns.color_palette("tab10", n_colors=len(versions))

    for i, version in enumerate(versions):
        group = df[df['Solver_Version'] == version]
        # Sort samples descending (Yield Curve)
        yields = sorted(group['Samples_Found'].tolist(), reverse=True)
        
        x_axis = range(1, len(yields) + 1)
        
        if version == BASELINE_NAME:
            plt.plot(x_axis, yields, color='black', linewidth=2.5, 
                     linestyle='--', label=version)
        else:
            plt.plot(x_axis, yields, color=palette[i], linewidth=2, alpha=0.8, label=version)

    plt.xlabel("Number of Benchmarks (Sorted by Yield)", fontsize=12)
    plt.ylabel("Samples Found", fontsize=12)
    plt.title("Sample Yield Profile (Throughput)", fontsize=14)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"throughput_curve.{OUTPUT_FORMAT}")
    print(f"Generated throughput_curve.{OUTPUT_FORMAT}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("main_csv", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--param", type=str, default="r")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_unify_data(args.main_csv, args.param, args.baseline)

    print_detailed_stats(df)
    
    # 1. Standard Time Plots
    plot_cdf_time(df, args.output_dir)
    
    # 2. New Partial Progress Plots
    plot_scatter_samples(df, args.output_dir)
    plot_throughput_curve(df, args.output_dir)

if __name__ == "__main__":
    main()
