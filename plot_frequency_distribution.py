# -*- coding: utf-8 -*-
"""
This script reads a file, counts the frequency of each unique line,
and plots a histogram of these frequencies.
"""

import argparse
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from subprocess import run
import sys


def plot_line_frequencies(file_path, num_samples, out_file):
    """
    Reads a file, calculates the frequency of each line, and plots the
    distribution of these frequencies.

    Args:
        file_path (str): The path to the input text file.
    """
    try:
        with Path(file_path).open('r') as f:
            line_counts = Counter(
                stripped_line
                for line in f
                # This is an assignment expression (walrus operator)
                if (stripped_line := line.strip())
            )
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    if not line_counts:
        print("The file is empty. Nothing to plot.")
        return

    # We are interested in the distribution of the counts themselves.
    # For example, how many lines appeared once, twice, etc.
    frequencies = line_counts.values()

    # --- Plotting the Histogram ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the histogram of the frequencies
    ax.hist(frequencies,  # type: ignore
            bins=range(1, max(frequencies) + 2),
            align='left', rwidth=0.8, color='skyblue', edgecolor='black')

    ax.set_title('Distribution of Samples', fontsize=16)
    ax.set_xlabel('Frequency (How many times a sample appeared)', fontsize=12)
    ax.set_ylabel('Number of Samples at this Frequency', fontsize=12)

    # Set integer ticks for the x-axis if the range is reasonable
    # if max(frequencies) < 20:
    ax.set_xticks(range(1, max(frequencies) + 1))

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a text box with summary statistics
    stats_text = (
        f"Total Samples: {num_samples}\n"
        f"Unique Samples: {len(line_counts)}\n"
        f"Most Frequent Samples appeared {max(frequencies)} times"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    print("Plot generated. Close the plot window to exit.")
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file)
        print(f"Plot saved to {out_file}")
    else:
        plt.show()


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Analyzes a file to plot the distribution of line frequencies.",
        epilog="Example: python line_frequency_plotter.py sample_data.txt"
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        help="The path to the text file to be analyzed.",
        default="samples.txt"
    )

    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        help="Total number of samples (lines) in the file.",
        default=1024
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output image file for the plot (e.g., histogram.png). If not provided, the plot will be shown interactively.",
        default=""
    )

    args = parser.parse_args()

    samples_file = Path(args.input)

    try:
      samples_file.unlink()
    except FileNotFoundError:
      pass

    # Somehow setting "-e 0.3" doesn't affect anything even in the original unigen
    # algo. It looks like it's setting the epsilon in ApproxMC, rather than that of
    # unigen.
    run(['./build/unigen_static', '-e', '0.3', '--verb', '2', '--samples', f"{args.num_samples}",
         '--sampleout', samples_file, 'benchmarks/test.cnf'])

    # Run the main function
    plot_line_frequencies(samples_file, args.num_samples, args.output)
