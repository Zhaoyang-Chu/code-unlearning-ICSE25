model=$1
peft_model=$2

accelerate launch  main.py \
    --model $model \
    --peft_model $peft_model \
    --tasks humaneval \
    --batch_size 20 \
    --max_length_generation 512 \
    --precision fp16 \
    --allow_code_execution \
    --metric_output_path $peft_model/humaneval_evaluation_results.json \
    --save_generations --save_generations_path $peft_model/humaneval_generations.json \
    --max_memory_per_gpu auto \
    --do_sample True \
    --temperature 0.2 \
    --top_p 0.95 \
    --n_samples 50 \
    --seed 42
