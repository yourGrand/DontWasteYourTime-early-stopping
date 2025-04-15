python e1.py plot \
    --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" "dynamic_adaptive_forgiving" \
    --outpath plots-category9-10 \
    --prefix results-main-mlp-10 \
    --n-splits 10 \
    --model mlp \
    --metric roc_auc_ovr \
    --time-limit 3600 \
    --kind speedups \
    data/mlp-nsplits-10-dynamic.parquet \

python e1.py plot \
    --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" "dynamic_adaptive_forgiving" \
    --outpath plots-category9-10 \
    --prefix results-main-rf-10 \
    --n-splits 10 \
    --model rf \
    --metric roc_auc_ovr \
    --time-limit 3600 \
    --kind speedups \
    data/rf-nsplits-10-dynamic.parquet \

python e1.py plot-stacked \
    --with-test \
    --methods "disabled" "current_average_worse_than_mean_best" "current_average_worse_than_best_worst_split" "dynamic_adaptive_forgiving" \
    --outpath plots-category9-10 \
    --prefix results-main \
    --n-splits 10 \
    --model mlp rf \
    --ax-height 5 \
    --ax-width 5 \
    --metric roc_auc_ovr \
    --time-limit 3600 \
    data/mlp-nsplits-10-dynamic.parquet data/rf-nsplits-10-dynamic.parquet \
