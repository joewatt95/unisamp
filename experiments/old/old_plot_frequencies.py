# This script reads a file line by line, counts the frequency of each unique line,
# and plots the results as a histogram.

from collections import Counter
from pathlib import Path
from subprocess import run
import matplotlib.pyplot as plt

samples_file = Path('samples.txt')
num_samples = 1024 * 10


def count_line_frequencies(filepath):
    """
    Reads a file line by line and returns a dictionary with each unique line as a key
    and its frequency (number of occurrences) as the value. This version processes
    the file using a memory-efficient iterator.

    Args:
        filepath (str): The path to the file to be read.

    Returns:
        dict: A dictionary mapping unique lines to their frequencies.
    """
    try:
        # Use a 'with' statement to ensure the file is automatically closed
        with Path(filepath).open('r') as file:
            line_counts = Counter(
                stripped_line
                for line in file
                # This is an assignment expression (walrus operator)
                if (stripped_line := line.strip())
            )

    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return dict(line_counts)


def plot_histogram(
        frequencies,
        title="Histogram of Line Frequencies",
        xlabel="Unique Lines",
        ylabel="Frequencies"):
  """
  Plots a histogram of the line frequencies using matplotlib.

  Args:
      frequencies (dict): A dictionary mapping unique lines to their frequencies.
  """
  if not frequencies:
    print("No data to plot.")
    return

  # Create the plot
  plt.figure(figsize=(10, 6))
  plt.bar(frequencies.keys(), frequencies.values(), color='skyblue')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()  # Adjust layout to prevent labels from overlapping
  plt.show()


# Main execution block
if __name__ == "__main__":
  try:
    samples_file.unlink()
  except FileNotFoundError:
    pass

  # Somehow setting "-e 0.3" doesn't affect anything even in the original unigen
  # algo. It looks like it's setting the epsilon in ApproxMC, rather than that of
  # unigen.
  run(['./build/unigen_static', '-e', '0.3', '--samples', f"{num_samples}",
       '--sampleout', samples_file, 'benchmarks/test.cnf'])
  # Call the function to count the lines in the sample file
  frequencies = count_line_frequencies(samples_file)

  # Plot the results
  if frequencies is not None:
    # print("\nLine Frequencies:")
    # Use a loop to print the dictionary contents in a readable format
    # for line, count in frequencies.items():
    #     print(f"'{line}': {count}")

    plot_histogram(frequencies, title=f"Histogram of all {num_samples} samples",
                   xlabel=f"Unique samples ({len(frequencies)})")
