hf_path="/rscratch/adityatomar/.hfcache/checkpoints/"
gradients_path="/rscratch/adityatomar/gradients/fisher_info"

#########################
# Llama-2-7b-hf
#########################

# model_id="meta-llama/Llama-2-7b-hf"
# model="Llama-2-7b-hf"

# CUDA_VISIBLE_DEVICES=0,1 python run-fisher.py \
#     --model_name_or_path $hf_path/$model_id \
#     --output_dir $gradients_path/$model_id \
#     --dataset wikitext2 \
#     --seqlen 2048 \
#     --maxseqlen 4096 \
#     --num_examples 16 > $model.log 2>&1 &


#########################
# Llama-2-13b-hf
#########################

model_id="meta-llama/Llama-2-13b-hf"
model="Llama-2-13b-hf"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run-fisher.py \
    --model_name_or_path $hf_path/$model_id \
    --output_dir $gradients_path/$model_id \
    --dataset wikitext2 \
    --seqlen 2048 \
    --maxseqlen 4096 \
    --num_examples 16 > $model.log 2>&1


#########################
# Llama-2-70b-hf
#########################

model_id="meta-llama/Llama-2-70b-hf"
model="Llama-2-70b-hf"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run-fisher.py \
    --model_name_or_path $hf_path/$model_id \
    --output_dir $gradients_path/$model_id \
    --dataset wikitext2 \
    --seqlen 2048 \
    --maxseqlen 4096 \
    --num_examples 16 > $model.log 2>&1


#########################
# Llama-3.1-8B
#########################

# model_id="meta-llama/Llama-3.1-8B"
# model="Llama-3.1-8B"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run-fisher.py \
#     --model_name_or_path $hf_path/$model_id \
#     --output_dir $gradients_path/$model_id \
#     --dataset wikitext2 \
#     --seqlen 2048 \
#     --maxseqlen 131072 \
#     --num_examples 16 > $model.log 2>&1


#########################
# Mistral-7B-v0.3
#########################

# model_id="mistralai/Mistral-7B-v0.3"
# model="Mistral-7B-v0.3"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run-fisher.py \
#     --model_name_or_path $hf_path/$model_id \
#     --output_dir $gradients_path/$model_id \
#     --dataset wikitext2 \
#     --seqlen 2048 \
#     --maxseqlen 32768 \
#     --num_examples 16 > $model.log 2>&1