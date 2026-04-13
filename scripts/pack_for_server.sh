#!/bin/bash
# Pack srdf-af project for upload to remote GPU server
# Usage: bash scripts/pack_for_server.sh
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACK_DIR="/tmp/srdf-af-pack"
REMOTE_HOST="root@connect.westd.seetacloud.com"
REMOTE_PORT="43296"
REMOTE_DATASET_DIR="~/autodl-fs/dataset"
REMOTE_PROJECT_DIR="~/autodl-fs/RLAIFNIG/srdf-af"
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR/srdf-af/data"

echo "=== 1. Copy project code ==="
cp -r "$PROJ_ROOT/srdf_af" "$PACK_DIR/srdf-af/"
cp -r "$PROJ_ROOT/configs" "$PACK_DIR/srdf-af/"
cp "$PROJ_ROOT/run.py" "$PACK_DIR/srdf-af/"
cp "$PROJ_ROOT/pyproject.toml" "$PACK_DIR/srdf-af/"

echo "=== 2. Copy R2R data + connectivity ==="
mkdir -p "$PACK_DIR/srdf-af/data/R2R"
cp /home/jwm/rlaifnig/Matterport3DSimulator/tasks/R2R/data/*.json \
   "$PACK_DIR/srdf-af/data/R2R/" 2>/dev/null || true
mkdir -p "$PACK_DIR/srdf-af/data/connectivity"
cp /home/jwm/rlaifnig/Matterport3DSimulator/connectivity/*.json \
   "$PACK_DIR/srdf-af/data/connectivity/" 2>/dev/null || true

echo "=== 3. Pack rendered images ==="
tar -cf "$PACK_DIR/images.tar" -C "$PROJ_ROOT/data" images/
echo "  $(du -sh $PACK_DIR/images.tar | cut -f1)"

echo "=== 4. Pack SFT checkpoints (optional) ==="
if [ -d "$PROJ_ROOT/output/round_0/speaker_v2" ]; then
    tar -cf "$PACK_DIR/speaker_v2.tar" -C "$PROJ_ROOT/output/round_0" speaker_v2/
    echo "  $(du -sh $PACK_DIR/speaker_v2.tar | cut -f1)"
fi

echo "=== 5. Create setup script ==="
cat > "$PACK_DIR/srdf-af/scripts/setup_server.sh" << 'SETUP'
#!/bin/bash
# Run this on the remote server after uploading
set -e

export HF_HOME="${HF_HOME:-$HOME/autodl-fs/.cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATA_ROOT="${DATA_ROOT:-$HOME/autodl-fs/dataset}"
mkdir -p "$DATA_ROOT" "$HOME/autodl-fs/output/srdf-af" "$HF_HOME"

echo "=== Install dependencies ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate peft bitsandbytes trl datasets
pip install qwen-vl-utils Pillow numpy opencv-python-headless tqdm fire pyyaml
pip install pycocoevalcap nltk

echo "=== Unpack images ==="
if [ -f ../images.tar ]; then
    mkdir -p "$DATA_ROOT"
    tar -xf ../images.tar -C "$DATA_ROOT/"
    echo "Images unpacked: $(find "$DATA_ROOT/images" -name '*.jpg' | wc -l) files"
fi

echo "=== Verify ==="
python -c "
from srdf_af.config import Config
cfg = Config.load('configs/pro6000.yaml')
print(f'Config loaded: speaker={cfg.speaker}, judge={cfg.judge_model}')
print(f'batch_size={cfg.batch_size}, max_images={cfg.max_images}')
import os
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
print(f'HF_HOME: {os.environ.get("HF_HOME")}')
"

echo "=== Ready! ==="
echo "Run: python run.py sft --config configs/pro6000.yaml"
echo " Or: python run.py flywheel --config configs/pro6000.yaml"
SETUP
chmod +x "$PACK_DIR/srdf-af/scripts/setup_server.sh"

echo ""
echo "=== Done ==="
echo "Pack directory: $PACK_DIR"
du -sh "$PACK_DIR"/*
echo ""
echo "Upload commands (preferred):"
echo "  rsync -avP -e 'ssh -p $REMOTE_PORT' $PACK_DIR/srdf-af/ $REMOTE_HOST:$REMOTE_PROJECT_DIR/"
echo "  rsync -avP -e 'ssh -p $REMOTE_PORT' $PACK_DIR/images.tar $REMOTE_HOST:$REMOTE_PROJECT_DIR/../"
echo "  # Dataset root on server: $REMOTE_DATASET_DIR"
echo "  # On server: cd $REMOTE_PROJECT_DIR && bash scripts/setup_server.sh"
echo ""
echo "Upload commands (fallback):"
echo "  scp -P $REMOTE_PORT -r $PACK_DIR/srdf-af $REMOTE_HOST:~/autodl-fs/RLAIFNIG/"
echo "  scp -P $REMOTE_PORT $PACK_DIR/images.tar $REMOTE_HOST:~/autodl-fs/RLAIFNIG/"
