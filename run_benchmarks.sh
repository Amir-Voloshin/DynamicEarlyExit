#!/bin/bash

# Define the values for datasets and criteria
# datasets=("cnn_dm_summarization" "xsum_summarization" "cnn_dm_lm" "human_eval")
datasets=("xsum_summarization" "cnn_dm_lm" "human_eval")
# datasets=("cnn_dm_summarization" "xsum_summarization")
# criteria=("convergence" "cosine_similarity" "token_repeat" "entropy_based" "entropy_gradient")
criteria=("convergence" "cosine_similarity" "token_repeat")

# Loop over datasets and criteria
for dataset in "${datasets[@]}"; do
  for criterion in "${criteria[@]}"; do
    echo "Running benchmark with dataset: $dataset and criteria: $criterion"
    
    python -m torch.distributed.run --nproc_per_node=1 benchmark.py \
      --model facebook/layerskip-llama3.2-1B \
      --criteria "$criterion" \
      --delta_threshold 0.1 \
      --dataset "$dataset" \
      --num_samples 35 \
      --generation_strategy autoregressive \
      --exit_layer 16 \
      --num_speculations 6 \
      --output_dir ./logs
    
    echo "Finished benchmark with dataset: $dataset and criteria: $criterion"
  done
done
