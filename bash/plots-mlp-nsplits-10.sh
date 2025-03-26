python e1.py plot \
    --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" \
    --outpath plots \
    --prefix results-main-mlp-10 \
    --n-splits 10 \
    --model mlp \
    --metric roc_auc_ovr \
    --time-limit 3600 \
    --kind speedups \
    data/mlp-nsplits-10.parquet \

python e1.py plot-stacked \
    --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" \
    --outpath plots \
    --prefix results-main \
    --n-splits 10 \
    --model mlp \
    --ax-height 3 \
    --ax-width 5 \
    --metric roc_auc_ovr \
    --time-limit 3600 \
    data/mlp-nsplits-10.parquet
