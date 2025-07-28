hf_path="/rscratch/adityatomar/.hfcache/checkpoints/"
gradients_path="/rscratch/adityatomar/gradients/fisher_info"

nsamples=16
seqlen=2048

#########################
# Llama-2-7b-hf
#########################

# model_id="meta-llama/Llama-2-7b-hf"
# model="Llama-2-7b-hf"

# mkdir -p $model

eval_dir="evals/wiki"
# mkdir -p $eval_dir

# bits=4

# CUDA_VISIBLE_DEVICES=2 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1 &

# bits=3

# CUDA_VISIBLE_DEVICES=3 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1 &

# bits=2

# CUDA_VISIBLE_DEVICES=4 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1 &

#########################
# Llama-2-13b-hf
#########################

model_id="meta-llama/Llama-2-13b-hf"
model="Llama-2-13b-hf"

mkdir -p $model

bits=4

CUDA_VISIBLE_DEVICES=2 python llama_simquant.py \
    $hf_path/$model_id \
    --abits $bits \
    --nsamples $nsamples \
    --seqlen $seqlen \
    --nuq \
    --include_sparse \
    --sparsity-threshold 0.99 \
    --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1 &

# bits=3

# CUDA_VISIBLE_DEVICES=4 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1 &

# bits=2

# CUDA_VISIBLE_DEVICES=5 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1

########################
# Llama-2-70b-hf
########################

# model_id="meta-llama/Llama-2-70b-hf"
# model="Llama-2-70b-hf"

# mkdir -p $model

# bits=4

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1

# bits=3

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1

# bits=2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1


#########################
# Llama-3.1-8B
#########################

# model_id="meta-llama/Llama-3.1-8B"
# model="Llama-3.1-8B"

# mkdir -p $model

# bits=4

# CUDA_VISIBLE_DEVICES=2 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1 &

# bits=3

# CUDA_VISIBLE_DEVICES=3 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1 &

# bits=2

# CUDA_VISIBLE_DEVICES=4 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1 &

#########################
# Mistral-7B-v0.3
#########################

# model_id="mistralai/Mistral-7B-v0.3"
# model="Mistral-7B-v0.3"

# mkdir -p $model

# bits=4

# CUDA_VISIBLE_DEVICES=5 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1 &

# bits=3

# CUDA_VISIBLE_DEVICES=6 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1 &

# bits=2

# CUDA_VISIBLE_DEVICES=7 python llama_simquant.py \
#     $hf_path/$model_id \
#     --abits $bits \
#     --nsamples $nsamples \
#     --seqlen $seqlen \
#     --nuq \
#     --include_sparse \
#     --sparsity-threshold 0.99 \
#     --quantizer-path ${model}/${bits}bit_quantizer.pickle > $eval_dir/${model}_${bits}bits.log 2>&1