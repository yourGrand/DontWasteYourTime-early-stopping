# Codebase Structure and Implementation

This document explains the structure and implementation of the codebase for early stopping in cross-validation experiments as described in the paper "Don't Waste Your Time: Early Stopping for Cross-Validation".

## Main Entry Point (e1.py)

### `cols_needed_for_plotting`

**Description**: Determines which columns are needed from experiment results for plotting.

**Parameters**:

- `metric`: The performance metric being evaluated
- `n_splits`: Optional parameter specifying number of CV splits

**Returns**:

- Core columns needed for all plots
- Metric-specific columns
- Split-specific columns for detailed analysis

### `exp_name_to_result_dir`

**Description**: Maps experiment names to their respective result directories.

**Implementation**: Uses pattern matching to organize different experiment categories and create appropriate storage paths for results.

### `experiment_set`

**Description**: The core configuration function that parameterises experiments.

**Parameters**:

- Experiment name that determines configuration settings

**Configuration Elements**:

- Default resources (time, memory, CPUs)
- Number of CV splits
- Early stopping methods to evaluate
- Pipeline type (MLP or RF classifier)
- Dataset selection criteria

**Returns**: A list of `E1` experiment objects configured according to the specified experiment name.

### `main`

**Description**: Entry point that handles command-line arguments and executes operations.

**Subcommands**:

- `run`: Execute experiments locally
- `submit`: Submit experiments to SLURM cluster
- `status`: Check experiment completion status
- `collect`: Gather results from completed experiments
- `plot`/`plot-stacked`: Generate visualisations of experiment results

## Experiment Implementation (exp1.py)

### E1 Class

**Description**: Main experiment class inheriting from Slurmable.

**Parameters**:

- Task (OpenML dataset ID)
- Fold number
- Metric to optimise
- Pipeline specification
- Early stopping strategy
- Resource requirements (CPUs, memory, time)

**Key Methods**:

- `get_data`: Retrieves and prepares dataset fold
- `history`: Loads stored experiment results
- `python_path`: Returns file path (for SLURM execution)

### Supporting Functions

#### `test_score_bagged_ensemble`

**Description**: Evaluates ensemble performance by aggregating models from cross-validation.

**Implementation**: Creates a voting classifier/regressor from trained models and calculates test scores on the ensemble.

#### `run_it`

**Description**: Main experiment execution function that orchestrates the experiment process.

**Implementation**:

- Sets up cross-validation splitter based on configuration
- Configures evaluation with early stopping if enabled
- Registers optimisation loop with the pipeline
- Executes the scheduler with specified time limits
- Saves results as parquet file
- Handles exceptions and resource cleanup

## Execution Flow

### 1. Experiment Setup

- User selects an experiment configuration (e.g., "reproduce", "category3-nsplits-10")
- `experiment_set` function generates multiple experiment variants with different parameters
- Each variant is instantiated as an `E1` object with specific configuration

### 2. Experiment Execution

- During execution:
  - Data is loaded from OpenML datasets
  - Cross-validation splitter is configured according to parameters
  - Early stopping method is applied based on configuration
  - Optimisation loop runs within specified time constraints
  - Results are saved systematically for later analysis

### 3. Result Analysis

- After experiment completion:
  - Results are collected using the `collect` command
  - Various visualisation plots are generated to analyse performance:
    - Incumbent traces showing performance progression over time
    - Ranking plots comparing different methods
    - Speedup calculations quantifying efficiency gains
