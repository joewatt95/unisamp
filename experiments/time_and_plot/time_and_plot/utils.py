"This module contains utility functions."

import csv
from typing import Any, DefaultDict
from .config import Config
from collections import defaultdict
import statistics


def setup_jobs(values: list[str], reps: int) -> list[dict[str, Any]]:
    """
    Creates a data structure that maps a job ID to its parameters.
    """
    job_id_to_info = []
    job_id_counter = 0
    for val in values:
        for i in range(reps):
            job_id_to_info.append(
                {"id": job_id_counter, "val": val, "rep": i + 1}
            )
            job_id_counter += 1
    return job_id_to_info


def calculate_statistics(
    job_results_by_val: DefaultDict[str, list]
) -> DefaultDict[str, dict]:
    """
    Calculates statistics (mean, median, min, max) for each value based on job results.
    """
    stats = defaultdict(dict)
    for val, results in job_results_by_val.items():
        timings = [t for t in results if t is not None]
        stats[val]["timings"] = timings
        if timings:
            stats[val]["mean"] = statistics.mean(timings)
            stats[val]["median"] = statistics.median(timings)
            stats[val]["min"] = min(timings)
            stats[val]["max"] = max(timings)
    return stats


def print_statistics_table(
    stats: DefaultDict[str, dict], sorted_values: list[str]
) -> None:
    """
    Prints the final statistics table to the console.
    """
    print("\n--- Final Statistics ---")
    print(
        f"{'Value':<20} {'Runs':<5} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10}"
    )
    print("-" * 67)

    for val in sorted_values:
        s = stats.get(val, {})
        timings = s.get("timings", [])
        if timings:
            print(
                f"{val:<20} {len(timings):<5} {s['mean']:<10.4f} {s['median']:<10.4f} {s['min']:<10.4f} {s['max']:<10.4f}"
            )
        else:
            print(
                f"{val:<20} {len(timings):<5} {'n/a':<10} {'n/a':<10} {'n/a':<10} {'n/a':<10}")


def write_csv_output(
    csv_filename: str,
    plot_values: list[str],
    stats: DefaultDict[str, dict[str, Any]],
    config: Config,
) -> None:
    """
    Writes the calculated statistics to a CSV file.
    """
    try:
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"# Command: {config.command}"])
            writer.writerow([f"# Repetitions: {config.repetitions}"])
            writer.writerow([f"# Parallel Workers: {config.parallel_workers}"])
            writer.writerow(["Value", "Runs", "Mean", "Median", "Min", "Max"])
            for val in plot_values:
                s = stats[val]
                writer.writerow(
                    [
                        val,
                        len(s["timings"]),
                        s["mean"],
                        s["median"],
                        s["min"],
                        s["max"],
                    ]
                )
        print(f"\nStatistics saved to '{csv_filename}'")
    except Exception as e:
        print(f"\nError writing to CSV file: {e}")
