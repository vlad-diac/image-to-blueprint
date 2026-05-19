#!/usr/bin/env bash
# install.sh — Full setup: Miniconda, blueprint env, pip deps, and model weights.
#
# Miniconda is installed into:  <repo>/.miniconda
# Environment name:             blueprint
# Models:                       <repo>/models/  (via download_models.sh)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# Optional:
#   HF_TOKEN=hf_xxx ./install.sh           (gated HuggingFace repos)
#   PYTORCH_CUDA=cu124 ./install.sh      (force PyTorch CUDA wheel tag)
#
# Activate after install:
#   source .miniconda/etc/profile.d/conda.sh
#   conda activate blueprint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${SCRIPT_DIR}/.miniconda"
ENV_NAME="blueprint"
PYTHON_VERSION="3.12"
INSTALLER="${SCRIPT_DIR}/Miniconda3-latest-Linux-installer.sh"

export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

# Override PyTorch CUDA wheel tag if auto-detection is wrong, e.g. PYTORCH_CUDA=cu124
PYTORCH_CUDA="${PYTORCH_CUDA:-}"

# -----------------------------------------------------------------------------
# CUDA / PyTorch detection
# -----------------------------------------------------------------------------

check_cuda() {
  echo "==> Checking NVIDIA GPU / CUDA driver ..."
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found — installing CPU-only PyTorch."
  else
    nvidia-smi || true
    echo ""
  fi
}

# Parse max CUDA version supported by the driver (e.g. 12.4) from nvidia-smi.
get_driver_cuda_version() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n 1
}

# Pick a PyTorch wheel index that matches (or is older than) the host driver.
# Installing a wheel built for newer CUDA than the driver supports causes:
#   RuntimeError: The NVIDIA driver on your system is too old
select_pytorch_index() {
  if [[ -n "${PYTORCH_CUDA}" ]]; then
    if [[ "${PYTORCH_CUDA}" == "cpu" ]]; then
      echo "https://download.pytorch.org/whl/cpu"
    else
      echo "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
    fi
    return 0
  fi

  local driver_cuda
  driver_cuda="$(get_driver_cuda_version || true)"
  if [[ -z "${driver_cuda}" ]]; then
    echo "https://download.pytorch.org/whl/cpu"
    return 0
  fi

  echo "==> Driver reports max CUDA: ${driver_cuda}" >&2

  # Compare major.minor as integers (bash-friendly).
  local major minor
  major="${driver_cuda%%.*}"
  minor="${driver_cuda#*.}"
  minor="${minor%%.*}"

  local tag
  if (( major < 12 || (major == 12 && minor < 1) )); then
    tag="cu118"
  elif (( major == 12 && minor < 4 )); then
    tag="cu121"
  elif (( major == 12 && minor < 6 )); then
    tag="cu124"
  elif (( major == 12 && minor < 8 )); then
    tag="cu126"
  else
    tag="cu128"
  fi

  echo "==> Selected PyTorch wheels: ${tag} (index: https://download.pytorch.org/whl/${tag})" >&2
  echo "https://download.pytorch.org/whl/${tag}"
}

install_pytorch() {
  local index_url
  index_url="$(select_pytorch_index)"

  echo "==> Installing PyTorch from ${index_url} ..."
  python -m pip install --upgrade pip
  python -m pip install --force-reinstall torch torchvision --index-url "${index_url}"
}

install_requirements_without_torch() {
  local req_filtered
  req_filtered="$(mktemp)"
  grep -vE '^(torch|torchvision)([=<>!~].*)?$' "${SCRIPT_DIR}/requirements.txt" > "${req_filtered}"
  python -m pip install -r "${req_filtered}"
  rm -f "${req_filtered}"
}

verify_cuda_torch() {
  echo "==> Verifying PyTorch / CUDA ..."
  python - <<'PY'
import sys

import torch

print(f"  torch:              {torch.__version__}")
print(f"  torchvision:        {__import__('torchvision').__version__}")
print(f"  torch.version.cuda: {torch.version.cuda}")
print(f"  cuda available:     {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    if torch.version.cuda:
        print("  [WARN] CUDA toolkit in wheel but torch.cuda.is_available() is False.")
    else:
        print("  [OK] CPU-only PyTorch (no CUDA in build).")
    sys.exit(0)

print(f"  device count:       {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  device {i}:           {torch.cuda.get_device_name(i)}")

try:
    x = torch.zeros(1, device="cuda")
    del x
    torch.cuda.synchronize()
    print("  cuda tensor test:   OK")
except Exception as exc:
    print(f"  cuda tensor test:   FAILED — {exc}", file=sys.stderr)
    print(
        "\nPyTorch CUDA build does not match this GPU driver.\n"
        "Re-run install with a matching wheel tag, e.g.:\n"
        "  PYTORCH_CUDA=cu124 ./install.sh\n"
        "  PYTORCH_CUDA=cu121 ./install.sh\n",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

# -----------------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------------

case "$(uname -m)" in
  x86_64|amd64)
    MINICONDA_ARCH="x86_64"
    ;;
  aarch64|arm64)
    MINICONDA_ARCH="aarch64"
    ;;
  *)
    echo "error: unsupported architecture: $(uname -m)" >&2
    echo "       supported: x86_64, aarch64" >&2
    exit 1
    ;;
esac

MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh"

# -----------------------------------------------------------------------------
# Download Miniconda
# -----------------------------------------------------------------------------

if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
  echo "==> Downloading Miniconda (${MINICONDA_ARCH}) ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${MINICONDA_URL}" -o "${INSTALLER}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "${MINICONDA_URL}" -O "${INSTALLER}"
  else
    echo "error: need curl or wget to download Miniconda" >&2
    exit 1
  fi

  echo "==> Installing Miniconda to ${CONDA_DIR} ..."
  bash "${INSTALLER}" -b -p "${CONDA_DIR}"
  rm -f "${INSTALLER}"
else
  echo "==> Miniconda already present at ${CONDA_DIR}"
fi

# shellcheck source=/dev/null
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# -----------------------------------------------------------------------------
# Accept conda Terms of Service (conda 24+)
# -----------------------------------------------------------------------------

echo "==> Accepting conda Terms of Service ..."
if conda tos --help >/dev/null 2>&1; then
  conda tos accept --all 2>/dev/null || true
  for channel in \
    "https://repo.anaconda.com/pkgs/main" \
    "https://repo.anaconda.com/pkgs/r" \
    "https://repo.anaconda.com/pkgs/msys2"; do
    conda tos accept --override-channels --channel "${channel}" 2>/dev/null || true
  done
fi

conda config --set auto_accept_tos yes 2>/dev/null || true

# -----------------------------------------------------------------------------
# Create environment
# -----------------------------------------------------------------------------

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Conda environment '${ENV_NAME}' already exists"
else
  echo "==> Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION}) ..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

# -----------------------------------------------------------------------------
# Python packages
# -----------------------------------------------------------------------------

check_cuda
install_pytorch
echo "==> Installing remaining Python packages ..."
install_requirements_without_torch
python -m pip install -U "huggingface_hub[cli]"
verify_cuda_torch

# -----------------------------------------------------------------------------
# Model weights
# -----------------------------------------------------------------------------

echo "==> Downloading models (this may take a while) ..."
bash "${SCRIPT_DIR}/download_models.sh"

echo ""
echo "Done."
echo ""
echo "  Conda root:   ${CONDA_DIR}"
echo "  Environment:  ${ENV_NAME} (Python ${PYTHON_VERSION})"
echo "  Models:       ${SCRIPT_DIR}/models/"
echo ""
echo "Activate:"
echo "  source ${CONDA_DIR}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Run the UI:"
echo "  python app.py"
