#!/bin/bash
# filepath: ~/DontWasteYourTime-early-stopping/bash/submit-all.sh

experiments=(
    # Category 3 - MLP pipeline
    "category3-nsplits-2-5"
    "category3-nsplits-20"
    "category3-nsplits-10"
    "category3-nsplits-5"
    "category3-nsplits-3"
    
    # Category 4 - RF pipeline
    "category4-nsplits-2-5"
    "category4-nsplits-20"
    "category4-nsplits-10"
    "category4-nsplits-5"
    "category4-nsplits-3"
    
    # Category 5 - SMAC with MLP
    "category5-nsplits-10"
    "category5-nsplits-20"
    
    # Category 6 - SMAC with RF
    "category6-nsplits-10"
    "category6-nsplits-20"
)

for exp in "${experiments[@]}"; do
  echo "Submitting experiment: $exp"
  python e1.py submit --expname "$exp" --overwrite-all --job-array-limit 1000
  
  sleep 1
done

echo "All experiments submitted!"
