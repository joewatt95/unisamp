"""This module handles the processing and output of results."""

import logging
import sys
from collections import defaultdict
from typing import Any, DefaultDict

from .config import Config
from .utils import calculate_statistics, print_statistics_table, write_csv_output

logger = logging.getLogger(__name__)


def process_and_output_results(
    config: Config,
    job_results_by_val: DefaultDict[str, list[float | None]],
) -> None:
    """
    Calculates final statistics, prints them, writes a CSV, and generates a plot.
    """
    logger.info("All jobs finished. Calculating statistics...")

    stats = calculate_statistics(job_results_by_val)

    try:
        sorted_values = sorted(config.values, key=float)
    except ValueError:
        sorted_values = sorted(config.values)

    print_statistics_table(stats, sorted_values)

    plot_values = [v for v in sorted_values if stats.get(v, {}).get("timings")]

    if not plot_values:
        logger.warning("No values were successfully timed. Exiting without plotting.")
        sys.exit(1)

    # Generate the plot
    if config.show_plot or config.output_plot:
        try:
            from .plotter import plot_results

            plot_means = [stats[v]["mean"] for v in plot_values]
            plot_results(
                plot_values,
                plot_means,
                config.repetitions,
                config.parallel_workers,
                config.command,
                show_plot=config.show_plot,
                output_plot=config.output_plot,
            )
        except ImportError:
            logger.error("'matplotlib' and 'numpy' libraries are required for plotting.")
            logger.error("Please install them using: poetry install --with plot")
        except Exception as e:
            logger.error(f"An error occurred during plotting: {e}")
            sys.exit(1)

    # --- CSV Output ---
    if config.output_stats:
        write_csv_output(config.output_stats, plot_values, stats, config)
