#!/bin/bash

### Baselines
# declare -A model_tasks=(
#     [llama-cnn]="cnn_dailymail"
#     [llama-wmt]="wmt16-ro-en"
#     [llama-mmlu]="mmlu"
#     [llama-piqa]="piqa"
# )

# for model in "${!model_tasks[@]}"
# do
#     model_path="/home/gstmchen/optimized-model-merging/llama/${model}/merged"
#     task="${model_tasks[$model]}"
#     echo $model_path
#     lm_eval --model vllm \
#             --model_args pretrained=$model_path,dtype=float16,tokenizer=meta-llama/Llama-2-7b-hf,gpu_memory_utilization=0.7,trust_remote_code=True \
#             --tasks $task \
#             --device cuda:1 \
#             --batch_size auto \
#             --output_path results \
#             --limit 500
# done

### Task-arithmetics & Ties
model_paths=(
task_arith tie
)

model_paths=("${model_paths[@]/#//home/gstmchen/optimized-model-merging/generative/outs/llama_merged/}")

for model_path in "${model_paths[@]}"
do
    echo $model_path
    CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
            --model_args pretrained=$model_path,dtype=float16,tokenizer=meta-llama/Llama-2-7b-hf,gpu_memory_utilization=0.4,trust_remote_code=True \
            --tasks mmlu \
            --device cuda:0 \
            --batch_size auto:4 \
            --output_path test \
            --limit 500
done

### Frank-Wolfe
# model_paths=(
# frank_wolfe_merge_005  frank_wolfe_merge_05  frank_wolfe_merge2_005  frank_wolfe_merge2_05
# frank_wolfe_merge_01   frank_wolfe_merge_1   frank_wolfe_merge2_01   frank_wolfe_merge2_1
# )

# model_paths=("${model_paths[@]/#//home/gstmchen/optimized-model-merging/generative/outs/llama_merged/frank_wolfe/}")

# for model_path in "${model_paths[@]}"
# do
#     echo $model_path
#     lm_eval --model vllm \
#             --model_args pretrained=$model_path,,dtype=float16,tokenizer=meta-llama/Llama-2-7b-hf,gpu_memory_utilization=0.7,trust_remote_code=True \
#             --tasks cnn_dailymail,mmlu,piqa,wmt16-ro-en \
#             --device cuda:1 \
#             --batch_size auto:4 \
#             --output_path results \
#             --log_samples \
#             --limit 500
# done

