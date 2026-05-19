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
# Optional: HF_TOKEN=hf_xxx ./install.sh  (gated HuggingFace repos)
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

echo "==> Installing Python packages ..."
python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
python -m pip install -U "huggingface_hub[cli]"

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
