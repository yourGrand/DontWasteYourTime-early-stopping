# Speedup Plots
# This shows how much faster the early stopping methods are compared to the baseline
python e1.py plot --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" \
  --outpath plots --prefix reproduce-results-10 \
  --n-splits 10 --model mlp --metric roc_auc_ovr \
  --time-limit 30 --kind speedups \
  data/reproduce-results.parquet

# Incumbent Traces Plots (with tests)
# To see how the incumbent changes over time for each method
python e1.py plot-stacked --with-test \
  --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" \
  --outpath plots --prefix reproduce-results \
  --n-splits 10 --model mlp --metric roc_auc_ovr \
  --time-limit 30 \
  data/reproduce-results.parquet

# Ranking Plots
# To see how methods rank against each other during optimisation
python e1.py plot --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" \
  --outpath plots --prefix reproduce-results-10 \
  --n-splits 10 --model mlp --metric roc_auc_ovr \
  --time-limit 30 --kind ranks-aggregated \
  data/reproduce-results.parquet