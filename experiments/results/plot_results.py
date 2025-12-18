#!/usr/bin/env python3
"""
plot_results.py

Features:
- **CDF Plot**: Replaces Cactus. Shows "Solved Instances (Y)" vs "CPU Time (X)".
- **Detailed Stats**: Computes PAR-2, Success Counts, and classifies Timeouts.
- **CPU Time Focus**: Uses scientific CPU time for all metrics.
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
 
    # Labeling
    if param_col in df_main.columns:
        df_main['Solver_Version'] = f"{param_col}=" + df_main[param_col].astype(str)
    else:
        df_main['Solver_Version'] = "New_Version"

    if baseline_csv:
        df_base = pd.read_csv(baseline_csv)
        df_base['Solver_Version'] = BASELINE_NAME
        df_final = pd.concat([df_main, df_base], ignore_index=True)
    else:
        df_final = df_main

    # --- TIME METRIC SELECTION ---
    # Prefer CPU_Time if available (new script), fallback to Wall_Time (old script legacy)
    if 'CPU_Time' in df_final.columns:
        time_metric = 'CPU_Time'
    elif 'Wall_Time' in df_final.columns:
        print("Warning: 'CPU_Time' not found. Using 'Wall_Time'. (Did you re-run collate?)")
        time_metric = 'Wall_Time'
    else:
        raise ValueError("No time column found in CSV")

    # Clean numeric
    df_final[time_metric] = pd.to_numeric(df_final[time_metric], errors='coerce')
    
    # Solved Logic
    # Old Logic (Too Strict):
    # df_final['Solved'] = df_final['Outcome'].str.contains('Success', ...

    # New Logic (CPU Focused):
    # Considers it solved if it says "Success" OR "Finished" (the Zombie case)
    def is_cpu_success(outcome):
        s = str(outcome).lower()
        # "Success" = clean exit
        # "Finished" = logic done, but maybe I/O timeout (Zombie)
        return "success" in s or "finished" in s

    df_final['Solved'] = df_final['Outcome'].apply(is_cpu_success)
    
    # PAR-2 & Capped Time Logic
    df_final['Time_PAR2'] = df_final[time_metric]
    df_final.loc[~df_final['Solved'], 'Time_PAR2'] = TIMEOUT * PAR_PENALTY
    
    df_final['Time_Capped'] = df_final[time_metric]
    df_final.loc[~df_final['Solved'], 'Time_Capped'] = TIMEOUT
    df_final.loc[df_final['Time_Capped'] > TIMEOUT, 'Time_Capped'] = TIMEOUT

    return df_final

def print_detailed_stats(df):
    print("\n" + "="*80)
    print(f"{'VERSION':<20} | {'PAR-2':<8} | {'SOLVED':<8} | {'TIMEOUTS (Breakdown)':<30}")
    print("="*80)

    for version, group in df.groupby('Solver_Version'):
        total = len(group)
        solved = group['Solved'].sum()
        par2 = group['Time_PAR2'].mean()
        
        # Breakdown Failures
        timeouts = group[group['Outcome'].str.contains('Timeout', na=False)]
        t_total = len(timeouts)
        
        # Try to classify timeouts if phase info exists "Timeout (Sampling)"
        # You might need to adjust string matching based on exact Outcome format
        t_samp = timeouts['Outcome'].str.contains('Sampling').sum()
        t_appmc = timeouts['Outcome'].str.contains('ApproxMC').sum()
        t_plain = t_total - t_samp - t_appmc
        
        # Format breakdown string
        breakdown = f"Tot:{t_total} (S:{t_samp}, A:{t_appmc}, ?: {t_plain})"
        
        print(f"{version:<20} | {par2:<8.1f} | {solved}/{total} | {breakdown}")
    print("="*80 + "\n")

def plot_cdf(df, output_dir):
    """
    Plots CDF-style: X-Axis = Time, Y-Axis = Count of Solved Instances.
    This replaces the inverted Cactus plot.
    """
    plt.figure(figsize=(10, 7))
    versions = sorted(df['Solver_Version'].unique())
    palette = sns.color_palette("tab10", n_colors=len(versions))

    for i, version in enumerate(versions):
        group = df[df['Solver_Version'] == version]
        
        # Get solved times and SORT them
        solved_times = sorted(group[group['Solved']]['Time_Capped'].tolist())
        
        if not solved_times: continue

        # X = Times
        # Y = Cumulative Count (1 to N)
        y_axis = range(1, len(solved_times) + 1)
        
        # Line Style
        if version == BASELINE_NAME:
            plt.plot(solved_times, y_axis, color='black', linewidth=2.5, 
                     linestyle='--', label=version, zorder=10)
        else:
            plt.plot(solved_times, y_axis, color=palette[i], 
                     linewidth=2, alpha=0.9, label=version)

    plt.xscale('log')
    plt.xlabel("CPU Time (s) [Log Scale]", fontsize=12)
    plt.ylabel("Number of Solved Instances (Cumulative)", fontsize=12)
    plt.title("Solved Instances over Time (CDF)", fontsize=14)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"cdf_comparison.{OUTPUT_FORMAT}", format=OUTPUT_FORMAT)
    print(f"Generated cdf_comparison.{OUTPUT_FORMAT}")

def plot_box_distributions(df, output_dir):
    """
    Box Plot: Shows runtime distribution for SOLVED instances only.
    - Hides outliers (showfliers=False) to prevent black bars of dots.
    - Uses Log Scale.
    """
    # 1. Filter: Only plot successfully solved instances
    solved_df = df[df['Solved']].copy()
    
    if solved_df.empty:
        print("Warning: No solved instances to plot boxes for.")
        return

    plt.figure(figsize=(12, 6))
    
    # 2. Ordering: Put Baseline last for easy comparison
    versions = sorted(solved_df['Solver_Version'].unique())
    if BASELINE_NAME in versions:
        versions.remove(BASELINE_NAME)
        versions.append(BASELINE_NAME)
    
    # 3. Plot
    # showfliers=False is CRITICAL for cleaning up 1900+ benchmark plots
    sns.boxplot(data=solved_df, x='Solver_Version', y='Time_Capped', 
                order=versions, palette="viridis", showfliers=False, linewidth=1)
    
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Solver Version", fontsize=12)
    plt.ylabel("CPU Time (s) [Log Scale]", fontsize=12)
    plt.title("Runtime Distribution (Median & Variance)", fontsize=14)
    plt.grid(True, axis='y', alpha=0.3, which='major')
    plt.tight_layout()
    
    out_file = output_dir / f"runtime_distributions.{OUTPUT_FORMAT}"
    plt.savefig(out_file, format=OUTPUT_FORMAT)
    print(f"Generated {out_file}")

def plot_scatter_best_vs_baseline(df, output_dir):
    """
    Scatter Plot: Compares Baseline vs. The Best New Configuration.
    - Points below diagonal = New Version is Faster.
    - Vertical Wall on right = Timeouts Rescued by New Version.
    """
    # 1. Sanity Check: Do we have a baseline?
    if BASELINE_NAME not in df['Solver_Version'].values:
        return

    # 2. Identify the "Best" New Version (Lowest PAR-2 Score)
    ranking = df.groupby('Solver_Version')['Time_PAR2'].mean().sort_values()
    best_new = None
    for version in ranking.index:
        if version != BASELINE_NAME:
            best_new = version
            break
            
    if not best_new:
        print("No new versions found to compare against baseline.")
        return
        
    print(f"Scatter Plot: Comparing '{BASELINE_NAME}' vs Best New '{best_new}'")

    # 3. Pivot Data for Head-to-Head
    # Rows=File, Cols=Version, Values=Time_Capped
    pivot = df.pivot(index='Input_File', columns='Solver_Version', values='Time_Capped')
    data = pivot[[BASELINE_NAME, best_new]].dropna()

    plt.figure(figsize=(7, 7))
    
    # 4. Plot
    # s=15 (small dots), alpha=0.4 (transparency) helps see density
    plt.scatter(data[BASELINE_NAME], data[best_new], 
                alpha=0.4, edgecolors='none', s=15, c='#1f77b4')
    
    # Diagonal "Tie" Line
    # Plot from min time (e.g. 0.01s) to max time (TIMEOUT)
    limit_min = min(data.min().min(), 0.1)
    limit_max = TIMEOUT
    plt.plot([limit_min, limit_max], [limit_min, limit_max], 'r--', linewidth=1.5, label="Equal Performance")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f"CPU Time: {BASELINE_NAME} (s)", fontsize=11)
    plt.ylabel(f"CPU Time: {best_new} (s)", fontsize=11)
    plt.title(f"Head-to-Head: Baseline vs {best_new}", fontsize=13)
    
    # Force axes to be square so the diagonal is actually 45 degrees
    plt.axis('square')
    plt.xlim([limit_min, limit_max * 1.1])
    plt.ylim([limit_min, limit_max * 1.1])
    
    plt.legend()
    plt.grid(True, which="major", alpha=0.3)
    plt.tight_layout()
    
    out_file = output_dir / f"scatter_best_vs_baseline.{OUTPUT_FORMAT}"
    plt.savefig(out_file, format=OUTPUT_FORMAT)
    print(f"Generated {out_file}")

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
    plot_cdf(df, args.output_dir)
    plot_scatter_best_vs_baseline(df, args.output_dir)
    plot_box_distributions(df, args.output_dir)

    # You can keep plot_scatter / plot_box here if desired

if __name__ == "__main__":
    main()