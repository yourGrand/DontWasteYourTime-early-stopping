# Methods Implementation

This document explains how the early stopping methods described in the paper "Don't Waste Your Time: Early Stopping for Cross-Validation" are implemented in the codebase.

## Early Stopping Methods

The paper introduces two main early stopping methods for cross-validation:

### 1. Aggressive Early Stopping (`CVEarlyStopCurrentAverageWorseThanMeanBest`)

**Paper Description**: This method stops evaluating a configuration as soon as its mean performance across the evaluated folds becomes worse than the mean performance of the best configuration seen so far. This corresponds to Equation (1) in the paper.

**Implementation**:

```python
class CVEarlyStopCurrentAverageWorseThanMeanBest:
    def __init__(self, metric: Metric):
        super().__init__()
        self.value_to_beat: float | None = None
        self.metric = metric

    def update(self, report: Trial.Report) -> None:
        if report.status is not Trial.Status.SUCCESS:
            return

        if self.value_to_beat is None:
            self.value_to_beat = self.mean_fold_value(report)
            return

        match self.metric.compare(
            v1=report.values[self.metric.name],
            v2=self.value_to_beat,
        ):
            case Metric.Comparison.BETTER:
                self.value_to_beat = self.mean_fold_value(report)
            case _:
                pass

    def mean_fold_value(self, report: Trial.Report) -> float:
        suffix = f"val_{self.metric.name}"
        scores = [v for k, v in report.summary.items() if k.endswith(suffix)]
        return float(np.mean(scores))

    def should_stop(
        self,
        trial: Trial,  # noqa: ARG002
        scores: CVEvaluation.SplitScores,
    ) -> bool:
        if self.value_to_beat is None:
            return False

        challenger = float(np.mean(scores.val[self.metric.name]))

        match self.metric.compare(v1=challenger, v2=self.value_to_beat):
            case Metric.Comparison.WORSE | Metric.Comparison.EQUAL:
                return True
            case Metric.Comparison.BETTER:
                return False
            case _:
                raise ValueError("Invalid comparison")
```

**Class Functions**:

- `update`: Updates the benchmark value if a new best configuration is found. Only configurations with better performance than the current best will update the benchmark
- `mean_fold_value`: Helper method that calculates the mean validation score across all folds for a configuration
- `should_stop`: Core decision function that compares the mean performance of current folds against the benchmark and returns `True` to stop evaluation if performance is worse

**Variables**:

- `self.value_to_beat`: The mean fold value of the best configuration seen so far
- `challenger`: The mean value of the currently evaluated folds for this configuration

### 2. Forgiving Early Stopping (`CVEarlyStopCurrentAverageWorseThanBestWorstSplit`)

**Paper Description**: This method only stops evaluating a configuration if its mean performance is worse than the performance of the worst fold of the best configuration seen so far. This is a more conservative approach corresponding to Equation (2) in the paper.

**Implementation**:

```python
class CVEarlyStopCurrentAverageWorseThanBestWorstSplit:
    def __init__(self, metric: Metric):
        super().__init__()
        self.metric = metric
        self.value_to_beat: float | None = None

    def update(self, report: Trial.Report) -> None:
        if report.status is not Trial.Status.SUCCESS:
            return

        if self.value_to_beat is None:
            self.value_to_beat = self.worst_fold_value(report)
            return

        match self.metric.compare(
            v1=report.values[self.metric.name],
            v2=self.value_to_beat,
        ):
            case Metric.Comparison.BETTER:
                self.value_to_beat = self.worst_fold_value(report)
            case _:
                pass

    def worst_fold_value(self, report: Trial.Report) -> float:
        suffix = f"val_{self.metric.name}"
        scores = (v for k, v in report.summary.items() if k.endswith(suffix))
        worst = max(scores) if self.metric.minimize else min(scores)
        return float(worst)

    def should_stop(
        self,
        trial: Trial,  # noqa: ARG002
        scores: CVEvaluation.SplitScores,
    ) -> bool:
        if self.value_to_beat is None:
            return False

        challenger = float(np.mean(scores.val[self.metric.name]))

        match self.metric.compare(v1=challenger, v2=self.value_to_beat):
            case Metric.Comparison.WORSE | Metric.Comparison.EQUAL:
                return True
            case Metric.Comparison.BETTER:
                return False
            case _:
                raise ValueError("Invalid comparison")
```

**Class Functions**:

- `update`: Updates the benchmark value when a new best configuration is found, but uses the worst fold's performance as the new benchmark
- `worst_fold_value`: Helper method that identifies the worst-performing fold from a configuration's results (maximum value if minimising, minimum if maximising)
- `should_stop`: Decision function that compares the mean performance of current folds against the worst fold performance of the best configuration

**Variables**:

- `self.value_to_beat`: The worst fold value of the best configuration seen so far (obtained from `worst_fold_value()` method)
- `challenger`: The mean value of the currently evaluated folds for this configuration

**Key Difference**: The implementations of both methods look similar, but they differ in what `self.value_to_beat` represents:

- For aggressive stopping: It's the mean of all fold scores for the best config
- For forgiving stopping: It's the worst individual fold score for the best config

### 3. Additional Implementation: `CVEarlyStopRobustStdOfTopN`

This method is implemented in the codebase but not used in the paper. It maintains a set of top N configurations and uses statistical bounds to determine if a new configuration has potential to be better than any in the current top N.
