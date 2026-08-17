#!/bin/bash
#SBATCH -p gpu_sun
#SBATCH --account=sun_hs285_contrib
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J idrift_new
#SBATCH --array=0-6
#SBATCH -o /n/scratch/users/a/alg7274/idrift/new_%A_%a.out
set -e
# Newest tested set (registered by o2_create_newest.sbatch). Each fits one L40S (48G).
MODELS=(gemma4:e4b gemma4:12b gemma4:31b phi4:mini phi4:14b qwen3.5:27b-q4 mistral-small:24b)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
PORT=$((11520 + SLURM_ARRAY_TASK_ID))
export OLLAMA_MODELS=/n/scratch/users/a/alg7274/ollama_models
export OLLAMA_HOST=127.0.0.1:$PORT
SAFE=$(echo "$MODEL" | tr ':/' '__')
echo "model=$MODEL port=$PORT gpu=$CUDA_VISIBLE_DEVICES host=$(hostname)"

~/ollama/bin/ollama serve > /n/scratch/users/a/alg7274/idrift/serve_new_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
SPID=$!
sleep 15
~/ollama/bin/ollama run "$MODEL" "hi" --keepalive 120m > /dev/null 2>&1 || echo "warmup nonzero (continuing)"

python3 ~/idrift/o2_infer.py \
  --exposure ~/idrift/exposure_full.jsonl \
  --out /n/scratch/users/a/alg7274/idrift/outputs_new_${SAFE}.jsonl \
  --model "$MODEL" --depth postedit --temperature 0.0 \
  --host http://127.0.0.1:$PORT

kill $SPID 2>/dev/null || true
echo "DONE $MODEL rows=$(wc -l < /n/scratch/users/a/alg7274/idrift/outputs_new_${SAFE}.jsonl)"
