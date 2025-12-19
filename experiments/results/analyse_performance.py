#!/usr/bin/env python3

"""
analyze_performance.py (Clean Labels + Sol_Diff)

- Metric: CPU Time.
- New Column: 'Sol_Diff' (Difference in samples found).
- Comparisons:
    1. Solved vs Solved (Speedup)
    2. Solved vs Unsolved (Rescue) -> Label is just "Rescue"
    3. Unsolved vs Solved (Regression) -> Label is just "Hard Regression"
    4. Unsolved vs Unsolved (Partial Progress) -> Label is "Partial Improvement"
- Detailed Reporting including sub-timings.

python analyse_performance.py results_new.csv --baseline results_old.csv --out-improvements out/improvements.csv --out-regressions out/regressions.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

def load_data(csv_path, exclude_unsat=False):
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    
    # 1. Force CPU Time Priority
    if 'CPU_Time' in df.columns:
        df['Time_Used'] = pd.to_numeric(df['CPU_Time'], errors='coerce')
    elif 'Wall_Time' in df.columns:
        df['Time_Used'] = pd.to_numeric(df['Wall_Time'], errors='coerce')
    else:
        df['Time_Used'] = pd.to_numeric(df['Real_Time'], errors='coerce')

    # 2. Sanitize Solution Counts
    if 'Solutions_Found' in df.columns:
        df['Solutions_Found'] = pd.to_numeric(df['Solutions_Found'], errors='coerce').fillna(0).astype(int)
    else:
        df['Solutions_Found'] = 0

    # 3. Define Solved Status
    def is_effectively_solved(outcome):
        s = str(outcome).lower()
        is_success = "success" in s
        is_unsat = "unsat" in s
        
        if exclude_unsat and is_unsat:
            return False 
            
        return is_success or is_unsat

    df['Is_Solved'] = df['Outcome'].apply(is_effectively_solved)
    return df

def analyze(main_df, base_df, out_imp, out_reg, threshold):
    # Parameter detection
    std_cols = ['Input_File', 'Run_ID', 'Outcome', 'Time_Used', 'Is_Solved', 
                'CPU_Time', 'Wall_Time', 'Real_Time', 'Space_MB', 
                'Sampling_Case', 'ApproxMC_Time', 'Sampling_Time', 'Avg_Sample_Time',
                'Success_Prob', 'Solutions_Found']
                
    param_cols = [c for c in main_df.columns if c not in std_cols and c in ['r', 'e', 'samples']]
    
    if not param_cols:
        main_df['Version_Label'] = "New_Version"
        versions = ["New_Version"]
    else:
        main_df['Version_Label'] = main_df[param_cols].apply(
            lambda x: "_".join(f"{k}={v}" for k, v in x.items()), axis=1
        )
        versions = sorted(main_df['Version_Label'].unique())

    print(f"Comparing {len(versions)} versions against Baseline (Threshold: {threshold}s)...")
    
    improvements = []
    regressions = []

    for ver in versions:
        ver_df = main_df[main_df['Version_Label'] == ver].copy()
        merged = pd.merge(ver_df, base_df, on='Input_File', suffixes=('_new', '_base'), how='inner')
        
        for idx, row in merged.iterrows():
            base_solved = row['Is_Solved_base']
            new_solved = row['Is_Solved_new']
            base_time = row['Time_Used_base']
            new_time = row['Time_Used_new']
            
            # Solution Counts & Difference
            base_sol = int(row.get('Solutions_Found_base', 0))
            new_sol = int(row.get('Solutions_Found_new', 0))
            sol_diff = new_sol - base_sol

            is_zombie = "Zombie" in str(row['Outcome_new'])

            # FULL DETAILS CAPTURE
            details = {
                'Version': ver, 
                'Input_File': row['Input_File'],
                'Base_Time': base_time, 
                'New_Time': new_time,
                
                # New Stats Column
                'Sol_Diff': sol_diff,
                
                'Base_Stat': row['Outcome_base'], 
                'New_Stat': row['Outcome_new'],
                'Base_Prob': row.get('Success_Prob_base', ''),
                'New_Prob': row.get('Success_Prob_new', ''),
                'Base_Sol': base_sol,
                'New_Sol': new_sol,
                'Base_Case': row.get('Sampling_Case_base', ''),
                'New_Case': row.get('Sampling_Case_new', ''),
                'Base_AppMC': row.get('ApproxMC_Time_base', ''),
                'New_AppMC': row.get('ApproxMC_Time_new', ''),
                'Base_Samp': row.get('Sampling_Time_base', ''),
                'New_Samp': row.get('Sampling_Time_new', ''),
            }

            # --- CATEGORY 1: BOTH SOLVED (Time Comparison) ---
            if base_solved and new_solved:
                diff = new_time - base_time
                if new_time < (base_time - threshold):
                    type_label = "Speedup"
                    if is_zombie: type_label += " [Zombie]"
                    entry = details.copy()
                    entry.update({'Type': type_label, 'Time_Diff': diff})
                    improvements.append(entry)
                elif new_time > (base_time + threshold):
                    entry = details.copy()
                    entry.update({'Type': 'Performance Regression', 'Time_Diff': diff})
                    regressions.append(entry)

            # --- CATEGORY 2: MIXED (Rescue / Crash) ---
            elif not base_solved and new_solved:
                # Rescue (Clean label)
                type_label = "Rescue"
                if is_zombie: type_label += " [Zombie]"
                entry = details.copy()
                entry.update({'Type': type_label, 'Time_Diff': -99999})
                improvements.append(entry)
            
            elif base_solved and not new_solved:
                # Hard Regression (Clean label)
                entry = details.copy()
                entry.update({'Type': 'Hard Regression', 'Time_Diff': 99999})
                regressions.append(entry)

            # --- CATEGORY 3: BOTH TIMEOUT (Partial Progress Check) ---
            elif not base_solved and not new_solved:
                if new_sol > base_sol:
                    entry = details.copy()
                    entry.update({
                        'Type': 'Partial Improvement',
                        'Time_Diff': 0
                    })
                    improvements.append(entry)
                
                elif new_sol < base_sol:
                    entry = details.copy()
                    entry.update({
                        'Type': 'Partial Regression',
                        'Time_Diff': 0
                    })
                    regressions.append(entry)

    # --- Saving ---
    col_order = [
        'Version', 'Input_File', 'Type', 'Time_Diff', 'Sol_Diff',
        'Base_Time', 'New_Time', 'Base_Stat', 'New_Stat',
        'Base_Sol', 'New_Sol', 
        'Base_Prob', 'New_Prob', 
        'Base_Case', 'New_Case',
        'Base_AppMC', 'New_AppMC', 
        'Base_Samp', 'New_Samp'
    ]

    def save_df(data, filename, is_regression=False):
        if not data:
            print(f"No {'regressions' if is_regression else 'improvements'} found.")
            return

        df = pd.DataFrame(data)
        final_cols = [c for c in col_order if c in df.columns]
        df = df[final_cols]
        
        if is_regression:
            # Sort Regressions: Worst sample loss first (ascending Sol_Diff), then worst time diff
            df = df.sort_values(by=['Version', 'Sol_Diff', 'Time_Diff'], ascending=[True, True, False])
        else:
            # Sort Improvements: Best sample gain first (descending Sol_Diff), then best time diff (ascending)
            df = df.sort_values(by=['Version', 'Sol_Diff', 'Time_Diff'], ascending=[True, False, True])
            
        df.to_csv(filename, index=False)
        print(f"Found {len(df)} {'regressions' if is_regression else 'improvements'}. Saved to '{filename}'")

    save_df(improvements, out_imp, is_regression=False)
    save_df(regressions, out_reg, is_regression=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("new_csv", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out-improvements", type=Path, default=Path("improvements.csv"))
    parser.add_argument("--out-regressions", type=Path, default=Path("regressions.csv"))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--exclude-unsat", action="store_true", help="Treat UNSAT results as Unsolved")
    
    args = parser.parse_args()

    print("Loading datasets...")
    df_new = load_data(args.new_csv, args.exclude_unsat)
    df_old = load_data(args.baseline, args.exclude_unsat)

    analyze(df_new, df_old, args.out_improvements, args.out_regressions, args.threshold)

if __name__ == "__main__":
    main()