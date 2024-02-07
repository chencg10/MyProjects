# OS Timer Simulation

This code simulates an operating system timer and calculates the turnaround time for different timing protocols. It reads input data from a file, simulates the execution of processes using various scheduling algorithms, and calculates the mean turnaround time for each algorithm.

## Features

- Supports the following timing protocols:
  - First-Come, First-Served (FCFS)
  - Last-Come, First-Served Non-preemptive (LCFS-NP)
  - Last-Come, First-Served Preemptive (LCFS-P)
  - Shortest Job First Preemptive (SJF-P)
  - Round Robin (RR)

- Input data is read from a file specified in the command line.

- The code is implemented in Python and uses the following libraries:
  - `numpy` for numerical calculations
  - `tqdm` for progress bar visualization

## Usage

1. Ensure that you have Python installed on your system.

2. Clone the repository or download the code files.

3. Open a terminal or command prompt and navigate to the directory containing the code files.

4. Run the following command to execute the code:

   ```
   python main.py input_file_path
   ```

   Replace `input_file_path` with the path to your input file. The input file should be formatted as follows:
   - The first line should contain the number of processes.
   - Each subsequent line should contain the arrival time and computation time for a process, separated by commas.

5. The code will simulate the timing protocols and display the progress using a progress bar. Once the simulation is complete, the mean turnaround time for each algorithm will be printed.

## Input File Example

```
3
0, 5
2, 3
1, 6
```

This input file represents three processes with their arrival times and computation times.

## License

This code is released under the [MIT License](LICENSE). Feel free to use and modify it according to your needs.

## Acknowledgements

The code was developed by Chen Cohen Gershon.
