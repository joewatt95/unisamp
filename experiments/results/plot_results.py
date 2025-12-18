#!/usr/bin/env python3
"""
plot_results.py

Visualizes benchmark comparisons with "De-cluttered" professional styling.
Saves outputs as SVG for high-quality inclusion in LaTeX/Papers.

Usage:
    python plot_results.py results_new.csv --baseline results_old.csv --param r
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
OUTPUT_FORMAT = "svg"  # Vector format for papers
# =================================================

def load_and_unify_data(main_csv, param_col, baseline_csv=None):
    # (Same loading logic as before - omitted for brevity but fully preserved in execution)
    df_main = pd.read_csv(main_csv)
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

    # Clean & Normalize
    for col in ['Wall_Time', 'ApproxMC_Time', 'Sampling_Time']:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    df_final['Solved'] = df_final['Outcome'].str.contains('Success', na=False, case=False)
    
    # Calc Metrics
    df_final['Time_PAR2'] = df_final['Wall_Time']
    df_final.loc[~df_final['Solved'], 'Time_PAR2'] = TIMEOUT * PAR_PENALTY
    
    df_final['Time_Capped'] = df_final['Wall_Time']
    df_final.loc[~df_final['Solved'], 'Time_Capped'] = TIMEOUT
    df_final.loc[df_final['Time_Capped'] > TIMEOUT, 'Time_Capped'] = TIMEOUT

    return df_final

def plot_cactus(df, output_dir):
    """
    CLEANUP APPLIED: Removed markers to prevent clutter with 1900+ points.
    """
    plt.figure(figsize=(10, 7)) # Standard paper size
    
    versions = sorted(df['Solver_Version'].unique())
    
    # Use a high-contrast palette
    palette = sns.color_palette("tab10", n_colors=len(versions))
    
    for i, version in enumerate(versions):
        group = df[df['Solver_Version'] == version]
        solved_times = sorted(group[group['Solved']]['Time_Capped'].tolist())
        
        if not solved_times: continue
            
        x_axis = range(1, len(solved_times) + 1)
        
        if version == BASELINE_NAME:
            # Baseline: Thick Black Dashed
            plt.plot(x_axis, solved_times, color='black', linewidth=2.5, 
                     linestyle='--', label=version, zorder=10)
        else:
            # Others: Thin Solid Lines, NO MARKERS
            plt.plot(x_axis, solved_times, color=palette[i], 
                     linewidth=2, alpha=0.9, label=version)

    plt.yscale('log')
    plt.xlabel("Number of Solved Instances", fontsize=12)
    plt.ylabel("Time (s) [Log Scale]", fontsize=12)
    plt.title("Cactus Plot: Solver Performance", fontsize=14)
    plt.grid(True, which="both", alpha=0.2, linewidth=0.5)
    plt.legend(loc="lower right", framealpha=0.9) # Move legend out of the way
    plt.tight_layout()
    
    plt.savefig(output_dir / f"cactus_comparison.{OUTPUT_FORMAT}", format=OUTPUT_FORMAT)
    print(f"Generated cactus_comparison.{OUTPUT_FORMAT}")

def plot_box_distributions(df, output_dir):
    """
    CLEANUP APPLIED: Disabled outliers (showfliers=False) to remove the black bar.
    """
    solved_df = df[df['Solved']].copy()
    if solved_df.empty: return

    plt.figure(figsize=(12, 6))
    
    versions = sorted(solved_df['Solver_Version'].unique())
    if BASELINE_NAME in versions:
        versions.remove(BASELINE_NAME)
        versions.append(BASELINE_NAME)
    
    # showfliers=False removes the thousands of dots at the top
    sns.boxplot(data=solved_df, x='Solver_Version', y='Time_Capped', 
                order=versions, palette="viridis", showfliers=False, linewidth=1)
    
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Solve Time (s) [Log Scale]", fontsize=12)
    plt.title("Runtime Distribution (Median & Variance)", fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / f"runtime_distributions.{OUTPUT_FORMAT}", format=OUTPUT_FORMAT)
    print(f"Generated runtime_distributions.{OUTPUT_FORMAT}")

def plot_scatter_best_vs_baseline(df, output_dir):
    """
    CLEANUP APPLIED: Reduced point size and alpha for dense clouds.
    """
    if BASELINE_NAME not in df['Solver_Version'].values: return

    ranking = df.groupby('Solver_Version')['Time_PAR2'].mean().sort_values()
    best_new = None
    for version in ranking.index:
        if version != BASELINE_NAME:
            best_new = version
            break
            
    if not best_new: return
        
    pivot = df.pivot(index='Input_File', columns='Solver_Version', values='Time_Capped')
    data = pivot[[BASELINE_NAME, best_new]].dropna()

    plt.figure(figsize=(7, 7))
    
    # Smaller dots (s=15), more transparency (alpha=0.4)
    plt.scatter(data[BASELINE_NAME], data[best_new], 
                alpha=0.4, edgecolors='none', s=15, c='#1f77b4')
    
    plt.plot([0.1, TIMEOUT], [0.1, TIMEOUT], 'r--', linewidth=1.5, label="Equal Performance")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f"Time (s): {BASELINE_NAME}", fontsize=11)
    plt.ylabel(f"Time (s): {best_new}", fontsize=11)
    plt.title(f"Comparison: {best_new} vs Baseline", fontsize=13)
    plt.legend()
    plt.grid(True, which="major", alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / f"scatter_best_vs_baseline.{OUTPUT_FORMAT}", format=OUTPUT_FORMAT)
    print(f"Generated scatter_best_vs_baseline.{OUTPUT_FORMAT}")

def print_leaderboard(df):
    print("\n" + "="*50)
    print(" LEADERBOARD (Ranked by PAR-2 Score)")
    print("="*50)
    
    stats = df.groupby('Solver_Version').agg(
        Solved=('Solved', 'sum'),
        Total=('Input_File', 'count'),
        PAR2=('Time_PAR2', 'mean'),
        Median_Time=('Time_Capped', 'median')
    ).sort_values('PAR2')
    
    stats['Solved %'] = (stats['Solved'] / stats['Total'] * 100).map('{:.1f}%'.format)
    stats['PAR2'] = stats['PAR2'].map('{:.1f}'.format)
    stats['Median_Time'] = stats['Median_Time'].map('{:.2f}'.format)
    
    print(stats[['Solved', 'Solved %', 'PAR2', 'Median_Time']])
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Plotter")
    parser.add_argument("main_csv", type=Path, help="CSV containing new results")
    parser.add_argument("--baseline", type=Path, help="CSV containing baseline results")
    parser.add_argument("--param", type=str, default="r", help="Parameter column to use as version label")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("plots"))
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_and_unify_data(args.main_csv, args.param, args.baseline)
        print_leaderboard(df)
        plot_cactus(df, args.output_dir)
        plot_box_distributions(df, args.output_dir)
        plot_scatter_best_vs_baseline(df, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
