#!/bin/bash
# Pack srdf-af project for upload to remote GPU server
# Usage: bash scripts/pack_for_server.sh
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACK_DIR="/tmp/srdf-af-pack"
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

echo "=== Install dependencies ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate peft bitsandbytes trl datasets
pip install qwen-vl-utils Pillow numpy opencv-python-headless tqdm fire pyyaml
pip install pycocoevalcap nltk

echo "=== Unpack images ==="
if [ -f ../images.tar ]; then
    mkdir -p data
    tar -xf ../images.tar -C data/
    echo "Images unpacked: $(find data/images -name '*.jpg' | wc -l) files"
fi

echo "=== Verify ==="
python -c "
from srdf_af.config import Config
cfg = Config.load('configs/pro6000.yaml')
print(f'Config loaded: speaker={cfg.speaker}, judge={cfg.judge_model}')
print(f'batch_size={cfg.batch_size}, max_images={cfg.max_images}')
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
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
echo "Upload commands:"
echo "  scp -r $PACK_DIR/srdf-af user@server:~/"
echo "  scp $PACK_DIR/images.tar user@server:~/"
echo "  # On server: cd srdf-af && bash scripts/setup_server.sh"
