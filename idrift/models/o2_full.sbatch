#!/bin/bash
#SBATCH -p gpu_sun
#SBATCH --account=sun_hs285_contrib
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J idrift_full
#SBATCH --array=0-3
#SBATCH -o /n/scratch/users/a/alg7274/idrift/full_%A_%a.out
set -e
# One array task per small instruct model; each fits on a single L40S (48G).
MODELS=(llama3.1:8b qwen2.5:7b gemma2:9b phi3:mini)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
PORT=$((11500 + SLURM_ARRAY_TASK_ID))   # unique port so co-located tasks don't clash
export OLLAMA_MODELS=/n/scratch/users/a/alg7274/ollama_models
export OLLAMA_HOST=127.0.0.1:$PORT
SAFE=$(echo "$MODEL" | tr ':/' '__')
echo "model=$MODEL port=$PORT gpu=$CUDA_VISIBLE_DEVICES host=$(hostname)"

~/ollama/bin/ollama serve > /n/scratch/users/a/alg7274/idrift/serve_full_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
SPID=$!
sleep 15
~/ollama/bin/ollama run "$MODEL" "hi" --keepalive 120m > /dev/null 2>&1 || echo "warmup nonzero (continuing)"

python3 ~/idrift/o2_infer.py \
  --exposure ~/idrift/exposure_full.jsonl \
  --out /n/scratch/users/a/alg7274/idrift/outputs_full_${SAFE}.jsonl \
  --model "$MODEL" --depth postedit --temperature 0.0 \
  --host http://127.0.0.1:$PORT

kill $SPID 2>/dev/null || true
echo "DONE $MODEL rows=$(wc -l < /n/scratch/users/a/alg7274/idrift/outputs_full_${SAFE}.jsonl)"
