# Innovative Step

The goal is to identify approaches that could potentially outperform both **“Aggressive”** and **“Forgiving”** strategies in terms of efficiency and/or effectiveness.

The ideal approach should:

1. Consider both mean performance and variance across folds.
2. Possibly adapt its behavior based on evaluation progress.
3. Maintain simplicity with minimal parameters.
4. Be applicable across both RF and MLP model types (unlike Robust-3/5 from the paper, which performed inconsistently and thus was not included in reported results).

---

## Current Early Stopping Methods

1. **Aggressive Early Stopping**: Stops cross-validation of a configuration if the mean score of evaluated folds is worse than or equal to the mean score of the incumbent (current best) configuration.

   $$
   E_{aggressive}(C_t, \{s^{c_j, i} | i \leq n\}) =
   \begin{cases}
   true, & \text{if mean-score}_n^{c_j} \leq \text{mean-score}_t^* \\
   false, & \text{otherwise,}
   \end{cases}
   $$

2. **Forgiving Early Stopping**: Stops cross-validation if the mean score of evaluated folds is worse than or equal to the worst individual fold score of the incumbent configuration.

   $$
   E_{aggressive}(C_t, \{s^{c_j, i} | i \leq n\}) =
   \begin{cases}
   true, & \text{if mean-score}_n^{c_j} \leq \text{worst-score}_t^* \\
   false, & \text{otherwise,}
   \end{cases}
   $$

The paper found that **Forgiving** generally outperformed **Aggressive** because the latter was too strict, often prematurely stopping configurations that might have been better if fully evaluated.

---

## Potential Directions for Improvement

1. **Dynamic Threshold Based on Fold Progress**:
   - The criteria could adjust based on how many folds have been evaluated.
   - Early in evaluation (fewer folds), being more forgiving avoids discarding potentially good configurations prematurely.
   - This can be expressed mathematically as:
     
     $$ \text{mean-score}_n^{c_j} \leq \text{worst-score}_t^* - \beta \times \frac{k - n}{k} $$
     
     where:
     - $k$ is the total number of folds,
     - $n$ is the number of completed folds,
     - $\beta$ is a small hyperparameter.

2. **Confidence-Based Early Stopping**:
   - Use a confidence-adjusted estimate of the configuration’s performance.

   $$ LCB = \text{mean-score}_n^{c_j} - \gamma \frac{\text{std-score}_n^{c_j}}{\sqrt{n}} $$

   - Stop if this lower confidence bound is below the incumbent’s worst fold:
     
     $$ LCB \leq \text{worst-score}_t^*  $$

3. **Trend-Based Early Stopping**:
   - Use a straightforward check on the trend of recent fold scores.
   - Compare the current fold to the previous one or the average of the previous few folds.
   - If the most recent fold’s performance is not recovering relative to earlier ones, this might be an additional signal to stop, even if the overall mean is close.
