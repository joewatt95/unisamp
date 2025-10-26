#!/usr/bin/env python3

"""
A script to time a given command with various input values and plot the results.

Example Usage:
python time_and_plot.py "sleep {}" 0.1 0.2 0.5 1.0
python time_and_plot.py -r 10 "my_program -n {}" 100 500 1000 5000
"""

import subprocess
import time
import argparse
import statistics
import shlex
import sys

# Try to import matplotlib. If it fails, provide a helpful error message.
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: The 'matplotlib' library is required to plot results.")
    print("Please install it, e.g., 'pip install matplotlib'")
    sys.exit(1)

# Default number of times to run the command for each value to get an average.
DEFAULT_REPS = 5


def time_command(command_template: str, value: str, reps: int) -> float | None:
    """
    Runs the command `reps` times, substituting `value` into the template.

    Returns the average execution time in seconds, or None if an error occurred.
    """
    timings = []
    print(f"  Testing value: {value} ({reps} repetitions)...")

    for i in range(reps):
        try:
            # 1. Format the command
            # Fills the placeholder {} with the current value
            full_command = command_template.format(value)

            # 2. Split the command string safely
            # shlex.split handles quotes and spaces correctly
            # e.g., "my_cmd -f 'file name'" becomes ['my_cmd', '-f', 'file name']
            command_args = shlex.split(full_command)

            # 3. Time the execution
            start_time = time.perf_counter()

            # Run the command.
            # check=True: Raises an error if the command fails (non-zero exit code)
            # capture_output=True: Hides the command's stdout/stderr from our console
            subprocess.run(
                command_args,
                check=True,
                capture_output=True,
                text=True
            )

            end_time = time.perf_counter()

            duration = end_time - start_time
            timings.append(duration)

        except subprocess.CalledProcessError as e:
            # Handle cases where the command itself fails
            print(
                f"    [Run {i+1}/{reps}] FAILED. Command returned non-zero exit code.")
            print(f"    Error: {e.stderr.strip()}")
            return None  # Signal failure for this value
        except FileNotFoundError:
            # Handle case where the command isn't found
            print(f"    Error: Command not found.")
            print(
                f"    Make sure '{command_args[0]}' is correct and in your system's PATH.")
            return None  # Signal failure
        except Exception as e:
            # Catch other potential errors
            print(f"    [Run {i+1}/{reps}] An unexpected error occurred: {e}")
            return None

    # 4. Calculate and return the average
    if timings:
        return statistics.mean(timings)
    else:
        return None


def plot_results(values: list[str], timings: list[float], command_template: str):
    """
    Generates and displays a plot of values vs. timings.
    """
    # Use the values as string-based categories for the x-axis
    # This correctly handles non-numeric inputs like "small", "medium", "large"
    x_axis_labels = values

    plt.figure(figsize=(10, 6))

    # Plot the data
    plt.plot(x_axis_labels, timings, marker='o', linestyle='-', color='b')

    # Add titles and labels
    plt.title(f"Command Execution Time Analysis", fontsize=16)
    plt.suptitle(f"Command: {command_template}", fontsize=10, y=0.92)
    plt.xlabel("Input Value", fontsize=12)
    plt.ylabel("Average Execution Time (seconds)", fontsize=12)

    # Add grid and layout adjustments
    plt.grid(True, linestyle='--', alpha=0.6)
    # Adjust next line for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # type: ignore

    # Save the plot to a file
    output_filename = "timing_plot.png"
    plt.savefig(output_filename)

    print(f"\nPlot saved as '{output_filename}'")

    # Display the plot
    plt.show()


def main():
    """
    Main function to parse arguments and orchestrate the timing and plotting.
    """
    parser = argparse.ArgumentParser(
        description="Run a command with different values, time it, and plot the results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Time the 'sleep' command with different durations
  python time_and_plot.py "sleep {}" 0.1 0.2 0.5 1.0

  # Time a program 'my_app' with different -n arguments, running 10 reps each
  python time_and_plot.py -r 10 "my_app -n {}" 100 500 1000 5000
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

    args = parser.parse_args()

    # Check if the placeholder is in the command
    if "{}" not in args.command:
        print(f"Error: Command template must include '{{}}' as a placeholder.")
        print(f"       You provided: \"{args.command}\"")
        sys.exit(1)

    print(f"Timing command: \"{args.command}\"")
    print(f"With {args.reps} repetitions for each value.\n")

    successful_values = []
    average_timings = []

    # Loop through each value provided by the user
    for val in args.values:
        avg_time = time_command(args.command, val, args.reps)

        if avg_time is not None:
            # Only add successful runs to our results
            print(f"  -> Average for {val}: {avg_time:.6f} seconds")
            successful_values.append(val)
            average_timings.append(avg_time)
        else:
            print(f"  -> Failed to get timing for value: {val}. Skipping.")

    # Check if we have any data to plot
    if not average_timings:
        print("\nNo successful runs were completed. Exiting without plotting.")
        sys.exit(1)

    # Generate the plot
    try:
        plot_results(successful_values, average_timings, args.command)
    except Exception as e:
        print(f"\nAn error occurred during plotting: {e}")
        print("Please ensure matplotlib is installed and working correctly.")


if __name__ == "__main__":
    main()
