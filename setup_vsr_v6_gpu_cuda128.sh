#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-vsr_v6_gpu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "错误：找不到 conda，请先初始化 Conda。" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "环境 ${ENV_NAME} 已存在。"
else
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

# The legacy requirements file contains CPU Paddle and PaddleOCR 2.x pins.
# Install the rest first, then install the V6 GPU stack explicitly.
sed \
    -e '/^paddleocr[<=>!~]/d' \
    -e '/^paddlepaddle[<=>!~]/d' \
    -e '/^paddlepaddle-gpu[<=>!~]/d' \
    -e '/^paddlex[<=>!~]/d' \
    -e '/^opencv-contrib-python[<=>!~]/d' \
    -e '/^opencv-python[<=>!~]/d' \
    -e '/^opencv-python-headless[<=>!~]/d' \
    -e '/^--extra-index-url/d' \
    "${SCRIPT_DIR}/requirements.txt" > "${TMPDIR:-/tmp}/vsr_v6_base_requirements.txt"

python -m pip install -r "${TMPDIR:-/tmp}/vsr_v6_base_requirements.txt"
python -m pip install "opencv-contrib-python==4.10.0.84"

python -m pip uninstall -y paddlepaddle paddlepaddle-gpu || true
python -m pip install "paddleocr==3.7.0" "paddlex==3.7.2"
python -m pip install \
    "paddlepaddle-gpu==3.3.0" \
    -i "https://www.paddlepaddle.org.cn/packages/stable/cu126/"

# CUDA 12.8 drivers can run the official CUDA 12.6 Paddle wheel.
# Install the PyTorch CUDA 12.8 build after Paddle. Its CUDA dependencies
# are resolved by the PyTorch wheel index.
python -m pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/cu128"

python - <<'PY'
import paddle
import paddleocr
import paddlex
import torch

print("paddle:", paddle.__version__)
print("paddle_cuda:", paddle.device.is_compiled_with_cuda())
print("paddleocr:", paddleocr.__version__)
print("paddlex:", getattr(paddlex, "__version__", "unknown"))
print("torch:", torch.__version__)
print("torch_cuda:", torch.cuda.is_available())

if not paddle.device.is_compiled_with_cuda():
    raise SystemExit("错误：Paddle 不是 CUDA 构建。")
if not torch.cuda.is_available():
    raise SystemExit("错误：Torch 无法使用 CUDA。")

paddle.set_device("gpu:0")
print("paddle_device:", paddle.get_device())
print("gpu:", torch.cuda.get_device_name(0))
PY

echo
echo "环境安装完成：${ENV_NAME}"
echo "启动命令："
echo "  conda activate ${ENV_NAME}"
echo "  cd ${SCRIPT_DIR}"
echo "  python backend/video_subtitle_remover_api_direct_v6.py"
