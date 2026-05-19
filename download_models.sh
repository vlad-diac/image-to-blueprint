#!/usr/bin/env bash
# download_models.sh — Download all models required by workflow.json
#
# Populates the ComfyUI models/ tree and a lightweight HF config snapshot so the
# standalone pipeline can run fully offline (set default_local in YAML).
#
# Usage:
#   ./download_models.sh
#   HF_TOKEN=hf_xxx ./download_models.sh
#
# Prerequisites (at least one):
#   pip install -U "huggingface_hub[cli]"   # preferred: hf download
#   wget or curl                            # fallback for direct URLs

set -euo pipefail

# =============================================================================
# CONFIG
# =============================================================================

# Repo root = directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_ROOT="${COMFY_ROOT:-$SCRIPT_DIR}"

MODELS_DIR="${COMFY_ROOT}/models"
UNET_DIR="${MODELS_DIR}/unet"
VAE_DIR="${MODELS_DIR}/vae"
TEXT_ENCODERS_DIR="${MODELS_DIR}/text_encoders"
LORAS_DIR="${MODELS_DIR}/loras"
SNAPSHOT_DIR="${MODELS_DIR}/Qwen--Qwen-Image-Edit-2511"
DEFAULT_REPO="Qwen/Qwen-Image-Edit-2511"

# Optional: set HF_TOKEN or run `huggingface-cli login` for gated repos
# export HF_TOKEN="hf_..."

# -----------------------------------------------------------------------------
# Models with URLs in workflow.json (auto-downloaded)
# -----------------------------------------------------------------------------

# CLIPLoader (node 93)
TEXT_ENCODER_REPO="Comfy-Org/HunyuanVideo_1.5_repackaged"
TEXT_ENCODER_REMOTE="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
TEXT_ENCODER_FILE="qwen_2.5_vl_7b_fp8_scaled.safetensors"

# VAELoader (node 95)
VAE_REPO="Comfy-Org/Qwen-Image_ComfyUI"
VAE_REMOTE="split_files/vae/qwen_image_vae.safetensors"
VAE_FILE="qwen_image_vae.safetensors"

# LoraLoaderModelOnly (node 102) — Lightning 4-step
LIGHTNING_REPO="lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_REMOTE="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LIGHTNING_FILE="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# -----------------------------------------------------------------------------
# Models NOT in workflow.json — fill in ONE of: REPO+REMOTE, or URL
# -----------------------------------------------------------------------------

# UnetLoaderGGUF (node 113)
GGUF_FILE="qwen-image-edit-2511-Q3_K_M.gguf"
# Option A: HuggingFace repo + path inside repo
GGUF_REPO=""                    # e.g. city96/Qwen-Image-Edit-2511-GGUF
GGUF_REMOTE=""                  # e.g. qwen-image-edit-2511-Q3_K_M.gguf
# Option B: direct URL (used if GGUF_REPO is empty)
GGUF_URL=""                     # e.g. https://huggingface.co/.../resolve/main/....gguf

# LoraLoaderModelOnly (node 109) — multiple angles
ANGLES_LORA_FILE="qwen-image-edit-2511-multiple-angles-lora.safetensors"
ANGLES_LORA_REPO=""             # e.g. your-username/qwen-multiple-angles-lora
ANGLES_LORA_REMOTE=""           # e.g. qwen-image-edit-2511-multiple-angles-lora.safetensors
ANGLES_LORA_URL=""              # direct URL alternative

# =============================================================================
# HELPERS
# =============================================================================

HF_AVAILABLE=0
WGET_AVAILABLE=0
CURL_AVAILABLE=0

check_prerequisites() {
    if command -v hf &>/dev/null; then
        HF_AVAILABLE=1
        echo "[OK] huggingface-cli (hf) found"
    else
        echo "[WARN] huggingface-cli (hf) not found — will use wget/curl where possible"
        echo "       Install: pip install -U \"huggingface_hub[cli]\""
    fi

    if command -v wget &>/dev/null; then
        WGET_AVAILABLE=1
        echo "[OK] wget found"
    elif command -v curl &>/dev/null; then
        CURL_AVAILABLE=1
        echo "[OK] curl found"
    else
        if [[ $HF_AVAILABLE -eq 0 ]]; then
            echo "[ERROR] Need hf, wget, or curl to download models."
            exit 1
        fi
    fi
    echo
}

# download_hf_file REPO REMOTE_PATH DEST_DIR DEST_FILENAME
# Downloads a single file from a HuggingFace repo into DEST_DIR/DEST_FILENAME.
download_hf_file() {
    local repo="$1"
    local remote_path="$2"
    local dest_dir="$3"
    local dest_filename="$4"

    mkdir -p "$dest_dir"
    local dest="${dest_dir}/${dest_filename}"

    if [[ -f "$dest" ]]; then
        echo "  [skip] already exists: $dest"
        return 0
    fi

    echo "  -> $dest"

    if [[ $HF_AVAILABLE -eq 1 ]]; then
        local tmp_dir
        tmp_dir="$(mktemp -d)"
        # shellcheck disable=SC2086
        if hf download "$repo" "$remote_path" \
            --local-dir "$tmp_dir"; then
            # hf may preserve remote subdirs under tmp_dir
            local found
            found="$(find "$tmp_dir" -type f -name "$dest_filename" | head -n 1)"
            if [[ -n "$found" && -f "$found" ]]; then
                mv "$found" "$dest"
                rm -rf "$tmp_dir"
                echo "  [done] $dest_filename"
                return 0
            fi
            # flat layout: file at tmp_dir/basename(remote_path)
            local basename
            basename="$(basename "$remote_path")"
            if [[ -f "$tmp_dir/$basename" ]]; then
                mv "$tmp_dir/$basename" "$dest"
                rm -rf "$tmp_dir"
                echo "  [done] $dest_filename"
                return 0
            fi
        fi
        rm -rf "$tmp_dir"
        echo "  [warn] hf download failed, trying wget/curl fallback..."
    fi

    local url="https://huggingface.co/${repo}/resolve/main/${remote_path}"
    download_url "$url" "$dest"
}

# download_url URL DEST_FILE
download_url() {
    local url="$1"
    local dest="$2"

    mkdir -p "$(dirname "$dest")"

    if [[ -f "$dest" ]]; then
        echo "  [skip] already exists: $dest"
        return 0
    fi

    echo "  -> $dest"

    if [[ $WGET_AVAILABLE -eq 1 ]]; then
        wget -q --show-progress -O "$dest" "$url"
    elif [[ $CURL_AVAILABLE -eq 1 ]]; then
        curl -# -L -o "$dest" "$url"
    else
        echo "  [ERROR] No download tool available for URL: $url"
        return 1
    fi

    echo "  [done] $(basename "$dest")"
}

# download_custom MODEL_LABEL DEST_DIR DEST_FILENAME REPO REMOTE URL
download_custom() {
    local label="$1"
    local dest_dir="$2"
    local dest_filename="$3"
    local repo="$4"
    local remote="$5"
    local url="$6"

    echo
    echo "--- $label ---"

    if [[ -n "$repo" && -n "$remote" ]]; then
        download_hf_file "$repo" "$remote" "$dest_dir" "$dest_filename"
        return $?
    fi

    if [[ -n "$url" ]]; then
        download_url "$url" "${dest_dir}/${dest_filename}"
        return $?
    fi

    echo "  [SKIP] No download source configured."
    echo "         Edit download_models.sh and set either:"
    echo "           - REPO + REMOTE path variables, or"
    echo "           - a direct URL variable"
    echo "         Then re-run this script."
    return 0
}

download_config_snapshot() {
    echo
    echo "============================================"
    echo "HF CONFIG SNAPSHOT (tokenizer / processor)"
    echo "============================================"
    echo "Repo:  $DEFAULT_REPO"
    echo "Dest:  $SNAPSHOT_DIR"
    echo "(weights excluded — ~5 MB)"
    echo

    mkdir -p "$SNAPSHOT_DIR"

    if [[ $HF_AVAILABLE -eq 0 ]]; then
        echo "[WARN] hf CLI required for config snapshot. Skipping."
        echo "       Install: pip install -U \"huggingface_hub[cli]\""
        return 0
    fi

    hf download "$DEFAULT_REPO" \
        --exclude "*.safetensors" \
        --exclude "*.bin" \
        --exclude "*.gguf" \
        --local-dir "$SNAPSHOT_DIR"

    echo
    echo "[done] Config snapshot at: $SNAPSHOT_DIR"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    echo "============================================"
    echo "Qwen Image Edit — Model Downloader"
    echo "============================================"
    echo "ComfyUI root: $COMFY_ROOT"
    echo

    check_prerequisites

    mkdir -p "$UNET_DIR" "$VAE_DIR" "$TEXT_ENCODERS_DIR" "$LORAS_DIR"

    echo "============================================"
    echo "WORKFLOW MODELS (from workflow.json)"
    echo "============================================"

    echo
    echo "--- Text Encoder (CLIPLoader) ---"
    download_hf_file "$TEXT_ENCODER_REPO" "$TEXT_ENCODER_REMOTE" \
        "$TEXT_ENCODERS_DIR" "$TEXT_ENCODER_FILE"

    echo
    echo "--- VAE (VAELoader) ---"
    download_hf_file "$VAE_REPO" "$VAE_REMOTE" \
        "$VAE_DIR" "$VAE_FILE"

    echo
    echo "--- Lightning LoRA (4-step) ---"
    download_hf_file "$LIGHTNING_REPO" "$LIGHTNING_REMOTE" \
        "$LORAS_DIR" "$LIGHTNING_FILE"

    echo
    echo "============================================"
    echo "CUSTOM MODELS (configure at top of script)"
    echo "============================================"

    download_custom "Transformer GGUF (UnetLoaderGGUF)" \
        "$UNET_DIR" "$GGUF_FILE" \
        "$GGUF_REPO" "$GGUF_REMOTE" "$GGUF_URL"

    download_custom "Multiple-angles LoRA (LoraLoaderModelOnly)" \
        "$LORAS_DIR" "$ANGLES_LORA_FILE" \
        "$ANGLES_LORA_REPO" "$ANGLES_LORA_REMOTE" "$ANGLES_LORA_URL"

    download_config_snapshot

    echo
    echo "============================================"
    echo "DOWNLOAD COMPLETE"
    echo "============================================"
    echo
    echo "Model layout:"
    echo "  ${UNET_DIR}/${GGUF_FILE}"
    echo "  ${VAE_DIR}/${VAE_FILE}"
    echo "  ${TEXT_ENCODERS_DIR}/${TEXT_ENCODER_FILE}"
    echo "  ${LORAS_DIR}/${LIGHTNING_FILE}"
    echo "  ${LORAS_DIR}/${ANGLES_LORA_FILE}"
    echo "  ${SNAPSHOT_DIR}/  (config snapshot)"
    echo
    echo "Standalone YAML — enable fully offline config lookups:"
    echo "  models:"
    echo "    default_local: models/Qwen--Qwen-Image-Edit-2511"
    echo
    echo "See standalone/tests/ship_blueprint_top.yaml for full example."
    echo
}

main "$@"
