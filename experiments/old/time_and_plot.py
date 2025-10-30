#!/usr/bin/env python3

"""
A script to time a command with various input values and display the results.

This script can run a user-provided command template against a list of
input values. It executes the command multiple times for each value to gather
timing statistics.

Features:
- Parallel execution of jobs using a thread pool.
- An interactive Text User Interface (TUI) powered by `curses` to show
  live progress and statistics (mean, median, min, max).
- A fallback simple console logger for terminals that do not support curses.
- Generation of a plot summarizing the mean execution times.
- Export of detailed statistics to a CSV file.
"""

import subprocess
import time
import argparse
import statistics
import shlex
import sys
import concurrent.futures
from collections import defaultdict
from pathlib import Path
import curses  # For TUI
from typing import Any, Callable, DefaultDict
import queue

# Try to import matplotlib and numpy. If it fails, provide a helpful error message.
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: 'matplotlib' and 'numpy' libraries are required to plot results.")
    print("Please install them, e.g., 'pip install matplotlib numpy'")
    print("(This script will exit after the TUI, as it cannot plot.)")
    # We don't exit here, as the TUI part can still run
    pass

# Default number of times to run the command for each value to get an average.
DEFAULT_REPS = 5


def run_single_command(
    job_id: int,
    status_queue: queue.Queue[dict[str, Any]],
    command_template: str,
    value: str
) -> None:
    """
    Runs a single instance of the command and puts its status into the queue.
    This function is designed to be executed in a separate thread.
    It does not print or interact with 'curses' directly.
    """
    try:
        # 1. Log when the worker picks up the job
        start_time = time.perf_counter()
        # No 'Running' status needed for this TUI, worker just... works.
        # status_queue.put({'id': job_id, 'status': 'Running', 'start_time': start_time})

        # Get the absolute path to the directory containing this script
        script_dir = Path(__file__).parent.resolve()

        # 2. Format and split the command
        full_command = command_template.format(value)
        command_args = shlex.split(full_command)

        # 3. Time the execution
        subprocess.run(
            command_args,
            check=True,
            capture_output=True,
            text=True,
            cwd=script_dir  # Set the working directory
        )

        end_time = time.perf_counter()
        duration = end_time - start_time

        # 4. Put final result in queue
        status_queue.put(
            {'id': job_id, 'status': 'Completed', 'duration': duration})

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip().split('\n')[-1]  # Get last line of error
        status_queue.put(
            {'id': job_id, 'status': 'Failed', 'error': error_msg})
    except FileNotFoundError:
        status_queue.put({'id': job_id, 'status': 'Failed',
                         'error': 'Command not found'})
    except Exception as e:
        status_queue.put({'id': job_id, 'status': 'Failed', 'error': str(e)})


def plot_results(
    values: list[str],
    means: list[float],
    command_template: str,
    output_filename: str | None
) -> None:
    """
    Generates, displays, and saves a plot of values vs. mean timings.

    Args:
        values: A list of the input values (x-axis).
        means: A list of the corresponding mean execution times (y-axis).
        command_template: The command that was run, for the plot title.
        output_filename: The path to save the plot image to. If None, the plot
                         is shown interactively instead.
    """
    x_axis_labels = values

    # Convert to numpy arrays for easier math
    mean_np = np.array(means)

    plt.figure(figsize=(10, 7))

    # Plot the mean data
    plt.plot(
        x_axis_labels,
        mean_np,
        '-o',  # Format: line ('-') with points ('o')
        label='Mean Execution Time'
    )

    # Add titles and labels
    plt.title(f"Command Execution Time Analysis", fontsize=16)
    plt.suptitle(f"Command: {command_template}", fontsize=10, y=0.92)
    plt.xlabel("Input Value", fontsize=12)
    plt.ylabel("Execution Time (seconds)", fontsize=12)

    # Add grid, legend, and layout adjustments
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    # Adjust for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore

    if output_filename:
        # Save the plot to a file and do not show it interactively.
        plt.savefig(output_filename)
        print(f"\nPlot saved as '{output_filename}'")
    else:
        # Display the plot interactively.
        print("\nDisplaying plot interactively...")
        plt.show()


def draw_value_tui(
    pad: curses.window,
    value_statuses: list[dict[str, Any]],
    pad_width: int,
    pad_scroll_pos: int
) -> None:
    """
    Performs the initial draw of all job status lines onto the TUI pad.

    Args:
        pad: The curses pad to draw on.
        value_statuses: A list of dictionaries, each representing the state
                        of a value being tested.
        pad_width: The width of the pad.
        pad_scroll_pos: The initial scroll position (should be 0).
    """
    # --- Job List ---
    for i, job in enumerate(value_statuses):
        draw_single_tui_line(pad, job, i, pad_width)


def draw_single_tui_line(
    pad: curses.window,
    job_status: dict[str, Any],
    index: int,
    pad_width: int
) -> None:
    """Draws a single line in the TUI pad for a given job status."""
    # Define colors (these are initialized in tui_main)
    COLOR_GREEN = curses.color_pair(1)
    COLOR_YELLOW = curses.color_pair(2)
    COLOR_RED = curses.color_pair(3)

    val = str(job_status['val'])
    reps_done = job_status['reps_done']
    reps_total = job_status['reps_total']
    timings = job_status['timings']

    # --- Progress Bar ---
    progress_pct = reps_done / reps_total
    bar_width = 10
    filled_len = int(bar_width * progress_pct)
    bar = '█' * filled_len + '-' * (bar_width - filled_len)
    progress_str = f"[{bar}] {reps_done}/{reps_total}"

    # --- Statistics ---
    mean_str = f"{statistics.mean(timings):.4f}" if timings else "n/a"
    median_str = f"{statistics.median(timings):.4f}" if timings else "n/a"
    min_str = f"{min(timings):.4f}" if timings else "n/a"
    max_str = f"{max(timings):.4f}" if timings else "n/a"

    # Truncate value if it's too long
    val_str = (val[:18] + '..') if len(val) > 20 else val

    line_str = f"{val_str:<20} {progress_str:<17} {mean_str:<10} {median_str:<10} {min_str:<10} {max_str:<10}"

    # --- Color ---
    color = curses.A_NORMAL
    if job_status['status'] == 'Pending':
        color = curses.A_DIM  # Use the A_DIM attribute directly
    elif job_status['status'] == 'Running':
        color = COLOR_YELLOW
    elif job_status['status'] == 'Completed':
        color = COLOR_GREEN
    elif job_status['status'] == 'Failed':  # If any rep failed
        color = COLOR_RED

    # Truncate the whole line to fit the pad width and add it to the pad
    line_to_draw = line_str.ljust(pad_width - 1)
    pad.addstr(index, 0, line_to_draw, color)


def run_all_jobs(
    args: argparse.Namespace,
    job_id_to_info: list[dict[str, Any]],
    update_handler: Callable[[dict[str, Any], DefaultDict[str, list]], None],
    should_continue: Callable[[], bool]
) -> DefaultDict[str, list]:
    """
    Core logic for submitting jobs and processing results.

    This function sets up a thread pool, submits all jobs, and then enters a
    This function sets up a thread pool, submits all jobs, and then enters a
    loop to process results from a queue. It uses a callback (`update_handler`)
    to allow different UIs (TUI or console) to process the results.

    Args:
        args: The parsed command-line arguments.
        job_id_to_info: A list mapping a job ID (index) to its metadata (value, rep).
        update_handler: A function to call with each status update from a job.
        should_continue: A function that returns False if the process should stop.

    Returns:
        A dictionary mapping each input value to a list of its timing results.
    """
    # This needs to be imported here because it's not at the top level
    import queue

    status_queue = queue.Queue()
    job_results_by_val = defaultdict(list)
    total_jobs = len(job_id_to_info)
    jobs_completed = 0

    # --- Executor and Job Submission ---
    # --- Executor Setup ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        for job in job_id_to_info:
            executor.submit(
                run_single_command,
                job['id'],
                status_queue,
                args.command,
                job['val'],
            )

        # --- Result Processing Loop ---
        while jobs_completed < total_jobs and should_continue():
            try:
                # Use a timeout to remain responsive to should_continue flag
                update = status_queue.get(timeout=0.1)
                jobs_completed += 1

                # Pass the update to the UI-specific handler
                update_handler(update, job_results_by_val)

            except queue.Empty:
                # No update, just continue the loop to check should_continue
                pass

    return job_results_by_val


def console_main(
    args: argparse.Namespace,
    job_id_to_info: list[dict[str, Any]]
) -> DefaultDict[str, list]:
    """
    Runs the experiment with simple print statements for progress.
    This is a fallback for when the TUI is disabled or unsupported.
    """
    print("TUI disabled. Using simple console logging.")
    print(
        f"Submitting {len(job_id_to_info)} jobs to a pool of {args.parallel or 'max'} workers...")

    # This flag allows the main loop to be interrupted by Ctrl-C
    keep_running = True

    def console_update_handler(
        update: dict[str, Any],
        job_results_by_val: DefaultDict[str, list]
    ) -> None:
        """Processes a single job update and prints it to the console."""
        job_info = job_id_to_info[update['id']]
        val, rep = job_info['val'], job_info['rep']
        total_done = sum(len(v) for v in job_results_by_val.values())

        if update['status'] == 'Completed':
            duration = update['duration']
            job_results_by_val[val].append(duration)
            print(
                f"  [Job OK] val={val} (rep {rep}): {duration:.4f}s ({total_done}/{len(job_id_to_info)} done)")
        elif update['status'] == 'Failed':
            error = update['error']
            job_results_by_val[val].append(None)  # Mark as failed
            print(
                f"  [Job FAILED] val={val} (rep {rep}): {error} ({total_done}/{len(job_id_to_info)} done)")

    try:
        job_results = run_all_jobs(
            args, job_id_to_info, console_update_handler, lambda: keep_running)
    except KeyboardInterrupt:
        print("\nCaught interrupt! Shutting down...")
        keep_running = False
        # The 'run_all_jobs' loop will exit, and the 'with' block will clean up.
        return defaultdict(list)  # Return empty results

    return job_results


def tui_main(
    stdscr: curses.window,
    args: argparse.Namespace
) -> DefaultDict[str, list]:
    """
    Sets up and runs the interactive TUI, orchestrating the job execution
    and display of live results. This function is wrapped by `curses.wrapper`.
    """

    # --- Curses Setup ---
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(True)  # Non-blocking getch
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)  # Completed
    curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Running
    curses.init_pair(3, curses.COLOR_RED, -1)  # Failed

    # --- Job and Queue Setup ---
    # This dict maps job_id -> index in value_statuses
    job_id_to_value_index = {}
    # This is our main TUI data structure, one entry PER VALUE
    value_statuses = []
    job_id_counter = 0

    for i, val in enumerate(args.values):
        value_statuses.append({
            'val': val,
            'status': 'Pending',
            'reps_done': 0,
            'reps_total': args.reps,
            'timings': [],
            'has_failed_reps': False
        })
        for _ in range(args.reps):
            job_id_to_value_index[job_id_counter] = i
            job_id_counter += 1

    # --- TUI State ---
    pad_scroll_pos = 0
    pad_width = 100  # Width of the scrollable pad
    keep_running = True  # Flag to control the main loop

    # Create a scrollable pad for just the job lines
    pad_height = len(value_statuses)
    try:
        pad = curses.newpad(pad_height, pad_width)
        pad.nodelay(True)
    except curses.error:
        # This can still fail if len(value_statuses) is enormous,
        raise Exception(
            f"Failed to create TUI pad. Too many values ({len(value_statuses)})?")

    # --- Initial Draw ---
    # Draw the static parts of the UI once
    stdscr.erase()
    stdscr.addstr(
        0, 0, f"Command: {args.command} (Press 'q' to quit, Up/Down to scroll)")
    stdscr.addstr(2, 0, "-" * (curses.COLS - 1))
    header = f"{'Value':<20} {'Progress':<17} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10}"
    stdscr.addstr(3, 0, header, curses.A_REVERSE)
    stdscr.addstr(4, 0, "-" * (curses.COLS - 1))

    pad.clear()
    # Draw all the initial job lines onto the pad
    draw_value_tui(pad, value_statuses, pad_width, pad_scroll_pos)

    status_queue = queue.Queue()
    job_results_by_val = defaultdict(list)
    total_jobs = job_id_counter
    jobs_completed = 0

    # --- TUI-specific Update Handler ---
    def tui_update_handler(
        update: dict[str, Any],
        updated_indices_set: set[int]
    ) -> None:
        """
        Processes a job update and updates the TUI data structures.
        """
        nonlocal jobs_completed
        job_id = update['id']
        value_index = job_id_to_value_index[job_id]
        val_stat = value_statuses[value_index]

        updated_indices_set.add(value_index)
        jobs_completed += 1
        val_stat['reps_done'] += 1
        val_stat['status'] = 'Running'

        if update['status'] == 'Completed':
            duration = update['duration']
            val_stat['timings'].append(duration)
            job_results_by_val[val_stat['val']].append(duration)
        elif update['status'] == 'Failed':
            val_stat['has_failed_reps'] = True
            val_stat['status'] = 'Failed'
            job_results_by_val[val_stat['val']].append(None)

        if val_stat['reps_done'] == val_stat['reps_total']:
            if not val_stat['has_failed_reps']:
                val_stat['status'] = 'Completed'

    # --- Executor and Job Submission ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        for job_id in range(total_jobs):
            value_index = job_id_to_value_index[job_id]
            val = value_statuses[value_index]['val']
            executor.submit(run_single_command, job_id,
                            status_queue, args.command, val)

        # --- TUI Event Loop ---
        while jobs_completed < total_jobs and keep_running:
            # Track which rows need redrawing in this frame
            updated_value_indices = set()

            # 1. Handle User Input
            try:
                screen_height, screen_width = stdscr.getmaxyx()
                page_size = screen_height - 5  # Number of visible lines in the pad
                max_scroll = max(0, pad_height - page_size)

                key = stdscr.getch()
                if key == ord('q'):
                    keep_running = False
                elif key == curses.KEY_DOWN:
                    pad_scroll_pos = min(pad_scroll_pos + 1, max_scroll)
                elif key == curses.KEY_UP:
                    pad_scroll_pos = max(pad_scroll_pos - 1, 0)
                elif key == curses.KEY_NPAGE:  # Page Down
                    pad_scroll_pos = min(
                        pad_scroll_pos + page_size, max_scroll)
                elif key == curses.KEY_PPAGE:  # Page Up
                    pad_scroll_pos = max(pad_scroll_pos - page_size, 0)
                elif key == curses.KEY_HOME:
                    pad_scroll_pos = 0
                elif key == curses.KEY_END:
                    pad_scroll_pos = max_scroll

            except curses.error:
                pass  # No input

            # 2. Process Status Updates from Queue
            while not status_queue.empty():
                try:
                    tui_update_handler(
                        status_queue.get_nowait(), updated_value_indices)
                except queue.Empty:
                    break

            # 3. Redraw UI
            # Only redraw the dynamic parts of the main screen header
            stdscr.addstr(
                1, 0, f"Progress: {jobs_completed}/{total_jobs} total reps done. Workers: {args.parallel or 'max'}.")

            # Tell curses we are about to update stdscr, but don't refresh the physical screen yet
            stdscr.noutrefresh()

            for index in updated_value_indices:
                # We draw to the pad regardless of whether the line is currently visible.
                # Curses' noutrefresh will handle clipping to the visible screen area.
                draw_single_tui_line(
                    pad, value_statuses[index], index, pad_width)

            # (pminrow, pmincol, sminrow, smincol, smaxrow, smaxcol)
            pad.noutrefresh(pad_scroll_pos, 0, 5, 0,
                            screen_height - 1, screen_width - 1)

            # Now, update the physical screen with all the changes at once.
            # This is the key to preventing flicker.
            curses.doupdate()

            time.sleep(0.05)  # Prevent high CPU usage from looping

    # --- End of Curses (wrapper will handle terminal restore) ---
    return job_results_by_val


def setup_jobs(values: list[str], reps: int) -> list[dict[str, Any]]:
    """
    Creates a data structure that maps a job ID to its parameters.

    Args:
        values: The list of input values to test.
        reps: The number of repetitions for each value.

    Returns:
        A list where the index is the job ID and the value is a dictionary
        containing the job's parameters (e.g., 'val', 'rep').
    """
    job_id_to_info = []
    job_id_counter = 0
    for val in values:
        for i in range(reps):
            job_id_to_info.append({
                'id': job_id_counter,
                'val': val,
                'rep': i + 1,
            })
            job_id_counter += 1
    return job_id_to_info


def process_and_output_results(
    args: argparse.Namespace,
    job_results_by_val: DefaultDict[str, list]
) -> None:
    """
    Calculates final statistics, prints them to the console, writes a CSV file,
    and generates the final plot.
    """
    print("\nAll jobs finished. Calculating statistics...")

    # --- Data Aggregation for Plotting and CSV ---
    stats = defaultdict(dict)
    # Try to sort values numerically if possible, otherwise alphabetically
    try:
        sorted_values = sorted(args.values, key=float)
    except ValueError:
        sorted_values = sorted(args.values)

    # --- Final Statistics Table ---
    print("\n--- Final Statistics ---")
    print(f"{'Value':<20} {'Runs':<5} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print("-" * 67)

    for val in sorted_values:
        results = job_results_by_val.get(val, [])
        stats[val]['timings'] = [t for t in results if t is not None]
        if stats[val]['timings']:
            stats[val]['mean'] = statistics.mean(stats[val]['timings'])
            stats[val]['median'] = statistics.median(stats[val]['timings'])
            stats[val]['min'] = min(stats[val]['timings'])
            stats[val]['max'] = max(stats[val]['timings'])
            print(
                f"{val:<20} {len(stats[val]['timings']):<5} {stats[val]['mean']:<10.4f} {stats[val]['median']:<10.4f} {stats[val]['min']:<10.4f} {stats[val]['max']:<10.4f}")
        else:
            print(f"{val:<20} {0:<5} {'n/a':<10} {'n/a':<10} {'n/a':<10} {'n/a':<10}")

    plot_values = [v for v in sorted_values if stats[v].get('timings')]
    if not plot_values:
        print("\nNo values were successfully timed. Exiting without plotting.")
        sys.exit(1)

    # Generate the plot
    try:
        plot_means = [stats[v]['mean'] for v in plot_values]
        plot_results(plot_values, plot_means, args.command, args.output)
    except NameError:
        print("\nCould not generate plot: 'matplotlib' or 'numpy' library was not found.")
    except Exception as e:
        print(f"\nAn error occurred during plotting: {e}")
        sys.exit(1)

    # --- CSV Output ---
    if args.csv:
        try:
            import csv
            with open(args.csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ['Value', 'Runs', 'Mean', 'Median', 'Min', 'Max'])
                for val in plot_values:
                    s = stats[val]
                    writer.writerow(
                        [val, len(s['timings']), s['mean'], s['median'], s['min'], s['max']])
            print(f"\nStatistics saved to '{args.csv}'")
        except Exception as e:
            print(f"\nError writing to CSV file: {e}")


def main() -> None:
    """Parses arguments and orchestrates the timing, UI, and plotting."""
    parser = argparse.ArgumentParser(
        description="Run a command with different values, time it, and plot the results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Time the 'sleep' command with different durations
  python time_and_plot.py "sleep {}" 0.1 0.2 0.5 1.0

  # Time a program 'my_app', running 10 reps each, using up to 4 cores
  python time_and_plot.py -r 10 -p 4 "my_app -n {}" 100 500 1000 5000
  
  # Save the plot to a specific file
  python time_and_plot.py -o my_custom_plot.svg "sleep {}" 0.1 0.2
"""
    )

    parser.add_argument(
        "command",
        type=str,
        help="The command template to run, with '{}' as the placeholder for the value."
    )

    parser.add_argument(
        "values",
        type=str,
        nargs='+',  # This gathers all remaining arguments into a list
        help="One or more values to substitute into the command placeholder."
    )

    parser.add_argument(
        "-r", "--reps",
        type=int,
        default=DEFAULT_REPS,
        help=f"Number of repetitions for each value to average. Default: {DEFAULT_REPS}"
    )

    parser.add_argument(
        "-p", "--parallel",
        type=int,
        default=None,
        help="Maximum number of parallel processes to use. Default: (all available cores)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename for the plot. If provided, the plot is saved to the file instead of being shown interactively."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output filename for the statistics in CSV format. If not provided, no CSV is generated."
    )

    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable the TUI and use simple console logging instead."
    )

    args = parser.parse_args()

    # --- Job Setup ---
    job_id_to_info = setup_jobs(args.values, args.reps)
    job_results_by_val = None

    # --- Main Execution ---
    if args.no_tui:
        # Run the simple console logger
        try:
            job_results_by_val = console_main(args, job_id_to_info)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)
    else:
        # Run the TUI logger
        try:
            job_results_by_val = curses.wrapper(tui_main, args)
        except curses.error:
            print("There was an error initializing the TUI.")
            print(
                "This can happen if the terminal is not supported (e.g., in some IDEs).")
            print("Try running again with the --no-tui flag.")
            sys.exit(1)
        except KeyboardInterrupt:
            # The TUI loop handles 'q' but not Ctrl-C, so wrapper catches it
            print("\nOperation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\nAn unexpected TUI error occurred: {e}")
            print("Please try again with the --no-tui flag.")
            sys.exit(1)

    # --- Final Output Processing ---
    # This part is now separated for clarity.
    process_and_output_results(args, job_results_by_val)


if __name__ == "__main__":
    main()
