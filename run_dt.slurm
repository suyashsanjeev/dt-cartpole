#!/bin/bash
#SBATCH --job-name=dt_cartpole
#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:02:00
#SBATCH --mem=16G
#SBATCH --output=dt_cartpole.%j.out
#SBATCH --error=dt_cartpole.%j.err
#SBATCH --export=ALL

# activate conda environment
source /gscratch/stf/ssuyash/miniconda/miniconda3/bin/activate
conda activate dt_env

# ensure conda’s C++ runtime is found
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


cd /gscratch/stf/ssuyash/project

# correct config check
echo "Python path: $(which python)"
python - <<'EOF'
import torch, sys, os
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
try:
    import stable_baselines3 as sb3
    print("SB3:", sb3.__version__)
except Exception as e:
    print("SB3 import failed:", e)
EOF


python decision_transformer_cartpole.py
