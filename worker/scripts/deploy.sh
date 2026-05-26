#!/usr/bin/env zsh
# deploy.sh — interactive RunPod network volume + provisioner pod setup
# Flow: pick GPU → pick datacenter that has it → create volume → create pod
# Usage: ./worker/scripts/deploy.sh
set -euo pipefail

BOLD=$'\033[1m'
DIM=$'\033[2m'
CYAN=$'\033[0;36m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
RED=$'\033[0;31m'
RESET=$'\033[0m'

header() { print "\n${BOLD}${CYAN}==> $*${RESET}"; }
ok()     { print "${GREEN}✓ $*${RESET}"; }
warn()   { print "${YELLOW}! $*${RESET}"; }
die()    { print "${RED}✗ $*${RESET}" >&2; exit 1; }

# numbered menu
# args: prompt label1 label2 ...
# sets PICK_INDEX (1-based) and PICK_LABEL
pick() {
  local prompt="$1"; shift
  local -a items=("$@")
  print "\n${BOLD}${prompt}${RESET}"
  local i=1
  for item in "${items[@]}"; do
    printf "  ${DIM}%2d)${RESET}  %s\n" "$i" "$item"
    (( i++ ))
  done
  local choice
  while true; do
    read "choice?
  Enter number: "
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#items[@]} )); then
      PICK_INDEX=$choice
      PICK_LABEL="${items[$choice]}"
      return
    fi
    warn "Invalid choice, try again."
  done
}

# ── 0. sanity check ───────────────────────────────────────────────────────────
command -v runpodctl &>/dev/null || die "runpodctl not found. Install: https://docs.runpod.io/runpodctl/install"
command -v jq        &>/dev/null || die "jq not found. Install: brew install jq"

# ── 1. fetch data ─────────────────────────────────────────────────────────────
header "Fetching GPU and datacenter data..."
GLOBAL_GPU_JSON=$(runpodctl gpu list 2>/dev/null || echo "[]")
DC_JSON=$(runpodctl datacenter list 2>/dev/null || echo "[]")
ok "Data fetched."

# ── 2. pick GPU ───────────────────────────────────────────────────────────────
# Build a deduplicated list of GPU IDs that appear in at least one datacenter,
# enriched with VRAM from the global list.

# Get all gpuIds that appear in any datacenter's gpuAvailability
local -a GPU_IDS GPU_LABELS
while IFS= read -r gpuId; do
  [[ -z "$gpuId" ]] && continue
  # skip duplicates
  if (( ${GPU_IDS[(Ie)$gpuId]} == 0 )); then
    vram=$(echo "$GLOBAL_GPU_JSON" | jq -r --arg g "$gpuId" \
      'first(.[] | select(.gpuId==$g) | .memoryInGb) // "?"')
    # count datacenters that list this GPU with non-empty stockStatus
    dc_count=$(echo "$DC_JSON" | jq -r --arg g "$gpuId" \
      '[.[] | select(.gpuAvailability[]?.gpuId==$g)] | length')
    GPU_IDS+=("$gpuId")
    GPU_LABELS+=("$gpuId  [${vram}GB VRAM — in $dc_count datacenter(s)]")
  fi
done < <(echo "$DC_JSON" | jq -r '.[] | .gpuAvailability[]?.gpuId')

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  warn "Could not parse GPU list. Enter GPU ID manually."
  print "  Hint: run 'runpodctl gpu list' to see all IDs."
  read "GPU_ID?  GPU ID: "
else
  pick "Select a GPU:" "${GPU_LABELS[@]}"
  GPU_ID="${GPU_IDS[$PICK_INDEX]}"
fi
ok "GPU: $GPU_ID"

# ── 3. pick datacenter that has this GPU ─────────────────────────────────────
header "Datacenters with $GPU_ID..."

local -a DC_IDS DC_LABELS
while IFS='|' read -r dcId location stockStatus; do
  [[ -z "$dcId" ]] && continue
  stock_label="${stockStatus:-unknown}"
  DC_IDS+=("$dcId")
  DC_LABELS+=("$dcId  ($location, stock: $stock_label)")
done < <(echo "$DC_JSON" | jq -r --arg g "$GPU_ID" \
  '.[] | . as $dc | (.gpuAvailability[]? | select(.gpuId==$g)) as $gpu |
   "\($dc.id)|\($dc.location // "")|\($gpu.stockStatus // "")"')

if [[ ${#DC_IDS[@]} -eq 0 ]]; then
  warn "No datacenters found for $GPU_ID. Enter datacenter ID manually."
  read "DATACENTER_ID?  Datacenter ID: "
else
  pick "Select a datacenter:" "${DC_LABELS[@]}"
  DATACENTER_ID="${DC_IDS[$PICK_INDEX]}"
fi
ok "Datacenter: $DATACENTER_ID"

# ── 4. volume config ──────────────────────────────────────────────────────────
header "Network Volume configuration"
read "VOL_NAME?  Volume name  [blueprint-vol]: "
VOL_NAME="${VOL_NAME:-blueprint-vol}"

read "VOL_SIZE?  Volume size GB [60]: "
VOL_SIZE="${VOL_SIZE:-60}"

# ── 5. optional extras ────────────────────────────────────────────────────────
read "REGISTRY_AUTH_ID?  Registry auth ID (leave blank to skip): "
read "HF_TOKEN?  HF_TOKEN for gated models (leave blank to skip): "

# ── 6. confirm ────────────────────────────────────────────────────────────────
print "\n${BOLD}Summary${RESET}"
print "  GPU          : ${CYAN}${GPU_ID}${RESET}"
print "  Datacenter   : ${CYAN}${DATACENTER_ID}${RESET}"
print "  Volume name  : ${CYAN}${VOL_NAME}${RESET}"
print "  Volume size  : ${CYAN}${VOL_SIZE} GB${RESET}"
[[ -n "$REGISTRY_AUTH_ID" ]] && print "  Registry auth: ${CYAN}${REGISTRY_AUTH_ID}${RESET}"
[[ -n "$HF_TOKEN" ]]         && print "  HF_TOKEN     : ${CYAN}(set)${RESET}"
print ""
read "confirm?Proceed? [y/N] "
[[ "$confirm" =~ ^[Yy]$ ]] || die "Aborted."

# ── 7. create volume ──────────────────────────────────────────────────────────
header "Creating network volume..."
VOL_OUT=$(runpodctl network-volume create \
  --name "$VOL_NAME" \
  --size "$VOL_SIZE" \
  --data-center-id "$DATACENTER_ID" \
  -o json 2>&1)

VOLUME_ID=$(echo "$VOL_OUT" | jq -r '.id // empty' 2>/dev/null || true)
if [[ -z "$VOLUME_ID" ]]; then
  warn "Could not parse volume ID from output:"
  print "$VOL_OUT"
  read "VOLUME_ID?  Paste the volume ID manually: "
fi
ok "Volume ID: $VOLUME_ID"

# ── 8. create provisioner pod ─────────────────────────────────────────────────
header "Creating provisioner pod..."

local -a POD_ARGS=(
  --network-volume-id "$VOLUME_ID"
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
  --gpu-id "$GPU_ID"
  --data-center-ids "$DATACENTER_ID"
  --name "blueprint-provisioner"
  --ports "22/tcp"
  --ssh
)

[[ -n "$REGISTRY_AUTH_ID" ]] && POD_ARGS+=(--registry-auth-id "$REGISTRY_AUTH_ID")
[[ -n "$HF_TOKEN" ]]         && POD_ARGS+=(--env "{\"HF_TOKEN\":\"${HF_TOKEN}\"}")

POD_OUT=$(runpodctl pod create "${POD_ARGS[@]}" -o json 2>&1)
POD_ID=$(echo "$POD_OUT" | jq -r '.id // empty' 2>/dev/null || true)

if [[ -z "$POD_ID" ]]; then
  warn "Could not parse pod ID from output:"
  print "$POD_OUT"
  POD_ID="<pod-id>"
fi
ok "Pod ID: $POD_ID"

# ── 9. next steps ─────────────────────────────────────────────────────────────
print "\n${BOLD}${GREEN}Done! Next steps:${RESET}"
print "  1. Wait for the pod to reach RUNNING state:"
print "     ${DIM}runpodctl pod list${RESET}"
print ""
print "  2. Once running, copy the worker scripts onto the pod:"
print "     ${DIM}scp -P <ssh-port> -r worker/ root@<pod-ip>:/workspace/worker/${RESET}"
print ""
print "  3. SSH in and run the provisioner:"
print "     ${DIM}ssh -p <ssh-port> root@<pod-ip>${RESET}"
print "     ${DIM}pip install huggingface_hub httpx${RESET}"
print "     ${DIM}python /workspace/worker/scripts/provision_volume.py${RESET}"
print ""
print "  4. Terminate the pod when done:"
print "     ${DIM}runpodctl pod terminate $POD_ID${RESET}"
print ""
print "  5. Save these for the serverless endpoint deploy:"
print "     ${BOLD}VOLUME_ID=${CYAN}${VOLUME_ID}${RESET}"
print "     ${BOLD}POD_ID   =${CYAN}${POD_ID}${RESET}"
