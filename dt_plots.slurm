#!/bin/bash
#SBATCH --job-name=dt_plots
#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --mem=4G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --export=ALL

# activate conda environment
source /gscratch/stf/ssuyash/miniconda/miniconda3/bin/activate
conda activate dt_env

# ensure conda’s C++ runtime is found
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /gscratch/stf/ssuyash/project

# config check
echo "Python path: $(which python)"
python - <<'EOF'
import torch, platform
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
EOF

# generate graphics
python target_vs_return.py
python return_histogram.py
python latent_space_map.py
python attention_heatmap_white.py
