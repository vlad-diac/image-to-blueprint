#!/usr/bin/env python3
"""
Copy the /workspace of one RunPod network volume into another (Method B: two pods).

A RunPod pod can attach only ONE network volume, and volumes are datacenter-bound,
so there is no single-pod `cp`. This script spins up one pod per volume (each mounts
its volume at /workspace), transfers pod-to-pod, verifies, then tears the pods down.

Transfer methods:
  rsync  (default) — rsync -avzP --inplace over SSH; resumable, efficient, best for
                     large data. The destination pod needs a public IP + exposed TCP.
  send             — `runpodctl send/receive` (croc P2P). No connectivity setup, but
                     no reliable resume; RunPod warns against it for large files.

Usage:
  # Clone into an existing destination volume
  python scripts/copy_network_volume.py --source-volume-id SRC --dest-volume-id DST

  # Create the destination volume first, then copy
  python scripts/copy_network_volume.py --source-volume-id SRC \
      --create-dest --dest-datacenter EU-RO-1 --size 60

  # croc instead of rsync; keep pods for debugging; dry run
  python scripts/copy_network_volume.py --source-volume-id SRC --dest-volume-id DST --method send
  python scripts/copy_network_volume.py --source-volume-id SRC --dest-volume-id DST --keep-pods
  python scripts/copy_network_volume.py --source-volume-id SRC --dest-volume-id DST --dry-run

Prerequisites:
  * runpodctl installed + authenticated (https://docs.runpod.io/runpodctl/install).
  * Your SSH public key registered with RunPod so `--ssh` injects it into the pods:
      runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub
    and your matching private key at --ssh-key (default: ~/.ssh/id_ed25519).
  * A GPU that exists in each volume's datacenter (default: "NVIDIA RTX 2000 Ada
    Generation"). Override with --gpu-id (see scripts/check_gpu_availability.py) or
    use --compute-type CPU.
  * If auto SSH detection fails, paste the pod's Connect-tab command via
    --source-ssh / --dest-ssh, e.g. --dest-ssh "root@1.2.3.4 -p 17445".

Nothing here runs inside a container — this is a local orchestration tool.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import secrets
import shlex
import subprocess
import sys
import time
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("copy_network_volume")

DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPU = "NVIDIA RTX 2000 Ada Generation"
DEFAULT_SSH_KEY = "~/.ssh/id_ed25519"
MOUNT = "/workspace"  # network volumes mount here on pods

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=1000",  # survive long, quiet rsync stretches
]

POD_RUNNING_TIMEOUT = 600   # seconds to reach RUNNING
SSH_READY_TIMEOUT = 300     # seconds for sshd to accept connections
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


# ── runpodctl helpers ─────────────────────────────────────────────────────────
def _run(cmd: list[str], timeout: int = 120,
         capture: bool = True, check: bool = False) -> subprocess.CompletedProcess:
    logger.debug("$ %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, text=True, timeout=timeout,
        capture_output=capture,
    )
    if check and proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() if capture else ""
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{err}")
    return proc


def _runpodctl_json(args: list[str], timeout: int = 120) -> Any:
    """Run `runpodctl <args> -o json` and parse, tolerating trailing log noise."""
    proc = _run(["runpodctl", *args, "-o", "json"], timeout=timeout, check=True)
    out = proc.stdout.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # some builds print a line before/after the JSON body
        start = min((i for i in (out.find("{"), out.find("[")) if i != -1), default=-1)
        end = max(out.rfind("}"), out.rfind("]"))
        if start != -1 and end > start:
            return json.loads(out[start:end + 1])
        raise


def _walk(obj: Any):
    """Yield every (key, value) pair anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def _first_value(obj: Any, key_names: set[str]) -> Any:
    for k, v in _walk(obj):
        if k in key_names and v not in (None, "", []):
            return v
    return None


# ── volumes ───────────────────────────────────────────────────────────────────
def resolve_volume_dc(volume_id: str, override: str | None) -> str:
    """Return the datacenter id for a volume (from --*-datacenter or volume list)."""
    if override:
        return override
    try:
        data = _runpodctl_json(["network-volume", "list"])
    except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Could not list network volumes to find the datacenter for {volume_id} "
            f"({exc}). Pass it explicitly with --source-datacenter/--dest-datacenter.",
        )
    items = data if isinstance(data, list) else data.get("networkVolumes") or data.get("data") or []
    for vol in items:
        if isinstance(vol, dict) and vol.get("id") == volume_id:
            dc = vol.get("dataCenterId") or vol.get("datacenterId") or _first_value(vol, {"dataCenterId", "datacenterId", "location"})
            if dc:
                return dc
    raise SystemExit(
        f"Volume {volume_id} not found in `runpodctl network-volume list`. "
        f"Pass its datacenter explicitly with --source-datacenter/--dest-datacenter.",
    )


def create_volume(name: str, size_gb: int, dc: str) -> str:
    logger.info("Creating destination volume %r (%d GB) in %s", name, size_gb, dc)
    data = _runpodctl_json([
        "network-volume", "create",
        "--name", name, "--size", str(size_gb), "--data-center-id", dc,
    ])
    vol_id = data.get("id") if isinstance(data, dict) else None
    vol_id = vol_id or _first_value(data, {"id"})
    if not vol_id:
        raise SystemExit(f"Could not parse new volume id from: {data!r}")
    logger.info("Created destination volume: %s", vol_id)
    return vol_id


# ── pod lifecycle ───────────────────────────────────────────────────────────
def create_pod(name: str, volume_id: str, dc: str, args: argparse.Namespace) -> str:
    cmd = [
        "runpodctl", "pod", "create",
        "--name", name,
        "--image", args.image,
        "--data-center-ids", dc,
        "--network-volume-id", volume_id,
        "--ports", "22/tcp",
        "--ssh",
        "--public-ip",
    ]
    if args.compute_type.upper() == "CPU":
        cmd += ["--compute-type", "CPU"]
    else:
        cmd += ["--gpu-id", args.gpu_id]
    if args.terminate_after:
        cmd += ["--terminate-after", args.terminate_after]
    cmd += ["-o", "json"]

    logger.info("Creating pod %r (volume %s @ %s)", name, volume_id, dc)
    proc = _run(cmd, timeout=180, check=True)
    try:
        data = json.loads(proc.stdout[proc.stdout.find("{"):proc.stdout.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        raise SystemExit(f"Could not parse pod id from create output:\n{proc.stdout}")
    pod_id = data.get("id") or _first_value(data, {"id"})
    if not pod_id:
        raise SystemExit(f"No pod id in create output:\n{proc.stdout}")
    logger.info("Created pod %s", pod_id)
    return pod_id


def get_pod(pod_id: str) -> Any:
    return _runpodctl_json(["pod", "get", pod_id])


def _status_of(info: Any) -> str:
    return str(
        _first_value(info, {"desiredStatus", "lastStatus", "status", "currentStatus"}) or "",
    ).upper()


def wait_pod_running(pod_id: str, timeout: int) -> Any:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            info = get_pod(pod_id)
        except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError):
            time.sleep(5)
            continue
        status = _status_of(info)
        if status != last:
            logger.info("Pod %s status: %s", pod_id, status or "?")
            last = status
        if status == "RUNNING":
            return info
        if status in ("EXITED", "TERMINATED", "FAILED", "DEAD"):
            raise SystemExit(f"Pod {pod_id} entered {status} before becoming RUNNING.")
        time.sleep(8)
    raise SystemExit(f"Pod {pod_id} did not reach RUNNING within {timeout}s.")


def delete_pod(pod_id: str) -> None:
    # subcommand naming varies across runpodctl versions; try the known spellings.
    for variant in (["pod", "terminate", pod_id], ["pod", "remove", pod_id],
                    ["pod", "delete", pod_id], ["remove", "pod", pod_id]):
        proc = _run(["runpodctl", *variant], timeout=60)
        if proc.returncode == 0:
            logger.info("Deleted pod %s (`runpodctl %s`)", pod_id, " ".join(variant))
            return
    logger.warning("Could not auto-delete pod %s — remove it manually.", pod_id)


# ── ssh ───────────────────────────────────────────────────────────────────────
class SshTarget:
    """How to reach a pod over SSH: direct (ip+port) or the ssh.runpod.io proxy."""

    def __init__(self, key: str, *, host: str, port: int | None = None,
                 user: str = "root", proxy: bool = False):
        self.key, self.host, self.port, self.user, self.proxy = key, host, port, user, proxy

    def base(self) -> list[str]:
        cmd = ["ssh", *SSH_OPTS, "-i", self.key]
        if self.port:
            cmd += ["-p", str(self.port)]
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    def __str__(self) -> str:
        port = f" -p {self.port}" if self.port else ""
        return f"{self.user}@{self.host}{port}"


def parse_ssh_override(spec: str, key: str) -> SshTarget:
    """Parse a pasted Connect-tab command like 'root@1.2.3.4 -p 17445' or 'x@ssh.runpod.io'."""
    spec = spec.strip()
    if spec.startswith("ssh "):
        spec = spec[4:].strip()
    port = None
    m = re.search(r"-p\s+(\d+)", spec)
    if m:
        port = int(m.group(1))
        spec = (spec[:m.start()] + spec[m.end():]).strip()
    m2 = re.search(r"-i\s+(\S+)", spec)
    if m2:
        key = m2.group(1)
        spec = (spec[:m2.start()] + spec[m2.end():]).strip()
    host_token = spec.split()[0]
    user, _, host = host_token.partition("@")
    if not host:
        user, host = "root", user
    return SshTarget(key, host=host, port=port, user=user, proxy=(host == "ssh.runpod.io"))


def resolve_ssh_target(pod_id: str, info: Any, key: str, override: str | None) -> SshTarget:
    if override:
        return parse_ssh_override(override, key)
    # 1) direct: public IP + the external port mapped to internal 22
    ip = _first_value(info, {"publicIp", "publicIP", "ip"})
    port = _find_ssh_port(info)
    if ip and _IPV4.match(str(ip)) and port:
        return SshTarget(key, host=str(ip), port=int(port))
    # 2) proxy: <podHostId>@ssh.runpod.io
    host_id = _first_value(info, {"podHostId", "podhostid"})
    if host_id:
        return SshTarget(key, host="ssh.runpod.io", user=str(host_id), proxy=True)
    raise SystemExit(
        f"Could not determine SSH details for pod {pod_id} from `runpodctl pod get`. "
        f"Re-run with --source-ssh/--dest-ssh using the pod's Connect-tab command.",
    )


def _find_ssh_port(info: Any) -> int | None:
    """Find the external port mapped to internal port 22 in a ports structure."""
    for _, v in _walk(info):
        if isinstance(v, dict):
            priv = v.get("privatePort") or v.get("internalPort")
            pub = v.get("publicPort") or v.get("externalPort")
            if str(priv) == "22" and pub:
                return int(pub)
    return None


def ssh_exec(target: SshTarget, remote_cmd: str, *, timeout: int = 300,
             stream: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [*target.base(), remote_cmd]
    proc = _run(cmd, timeout=timeout, capture=not stream, check=False)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if not stream else ""
        raise RuntimeError(f"remote command failed on {target}: {remote_cmd}\n{detail}")
    return proc


def wait_ssh(target: SshTarget, timeout: int) -> None:
    logger.info("Waiting for SSH on %s ...", target)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        proc = _run([*target.base(), "-o", "ConnectTimeout=10", "echo ready"],
                    timeout=30, capture=True)
        if proc.returncode == 0 and "ready" in proc.stdout:
            logger.info("SSH ready on %s", target)
            return
        time.sleep(6)
    raise SystemExit(f"SSH did not become ready on {target} within {timeout}s.")


# ── transfer ──────────────────────────────────────────────────────────────────
def _apt_install(target: SshTarget, pkgs: str) -> None:
    ssh_exec(target, f"command -v {pkgs.split()[0]} >/dev/null 2>&1 || "
                     f"(apt-get update -y && apt-get install -y {pkgs})",
             timeout=600)


def transfer_rsync(src: SshTarget, dst: SshTarget, dst_info: Any, timeout: int) -> None:
    logger.info("=== rsync transfer (source → destination) ===")
    _apt_install(src, "rsync openssh-client")
    _apt_install(dst, "rsync")

    # ephemeral keypair on the source; authorize it on the destination
    ssh_exec(src, "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
                  "test -f ~/.ssh/volcopy || ssh-keygen -t ed25519 -f ~/.ssh/volcopy -N '' -q")
    pub = ssh_exec(src, "cat ~/.ssh/volcopy.pub").stdout.strip().splitlines()[-1].strip()
    if not pub.startswith("ssh-"):
        raise SystemExit(f"Unexpected source public key: {pub!r}")
    ssh_exec(dst, "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
                  f"printf '%s\\n' {shlex.quote(pub)} >> ~/.ssh/authorized_keys && "
                  "chmod 600 ~/.ssh/authorized_keys")

    # destination's own public IP + external port for 22, straight from the pod
    dst_ip = ssh_exec(dst, "printenv RUNPOD_PUBLIC_IP").stdout.strip()
    dst_port = ssh_exec(dst, "printenv RUNPOD_TCP_PORT_22").stdout.strip()
    if not (_IPV4.match(dst_ip) and dst_port.isdigit()):
        # fall back to whatever we resolved for the local→dst connection
        dst_ip = dst.host if dst.port else dst_ip
        dst_port = str(dst.port or dst_port)
    if not (_IPV4.match(dst_ip) and str(dst_port).isdigit()):
        raise SystemExit(
            "Could not determine the destination pod's public IP/port for rsync. "
            "Ensure the dest pod has a public IP + exposed TCP 22, or use --method send.",
        )
    logger.info("Destination reachable at root@%s:%s", dst_ip, dst_port)

    inner_ssh = (f"ssh -p {dst_port} -i ~/.ssh/volcopy "
                 "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")
    rsync_cmd = (f"rsync -avzP --inplace --numeric-ids -e {shlex.quote(inner_ssh)} "
                 f"{MOUNT}/ root@{dst_ip}:{MOUNT}/")
    logger.info("Running: %s", rsync_cmd)
    ssh_exec(src, rsync_cmd, timeout=timeout, stream=True)
    logger.info("rsync completed.")


def transfer_send(src: SshTarget, dst: SshTarget, timeout: int) -> None:
    logger.info("=== runpodctl send/receive transfer ===")
    logger.warning("send/receive (croc) has no reliable resume; rsync is safer for large volumes.")
    for t in (src, dst):
        ssh_exec(t, "command -v runpodctl >/dev/null 2>&1 || "
                    "(echo 'runpodctl missing on pod' >&2; exit 1)")
    code = f"volcopy-{secrets.token_hex(4)}"
    # start the sender in the background so we can launch the receiver against the same code
    send_cmd = (f"cd {MOUNT} && nohup runpodctl send --code {code} . "
                ">/tmp/volcopy_send.log 2>&1 & echo started")
    ssh_exec(src, send_cmd, timeout=60)
    time.sleep(8)
    logger.info("Receiving on destination (code %s) ...", code)
    ssh_exec(dst, f"cd {MOUNT} && runpodctl receive {code}", timeout=timeout, stream=True)
    send_log = ssh_exec(src, "cat /tmp/volcopy_send.log 2>/dev/null || true", check=False).stdout
    logger.info("Sender log:\n%s", send_log.strip())


def verify(src: SshTarget, dst: SshTarget) -> None:
    logger.info("=== verifying (file count + total bytes) ===")
    probe = (f"find {MOUNT} -type f -printf '%s\\n' 2>/dev/null | "
             "awk '{c++; s+=$1} END{printf \"%d %d\\n\", c+0, s+0}'")
    s_out = ssh_exec(src, probe).stdout.strip()
    d_out = ssh_exec(dst, probe).stdout.strip()
    logger.info("source     : %s (files bytes)", s_out)
    logger.info("destination: %s (files bytes)", d_out)
    if s_out == d_out:
        logger.info("✓ counts and byte totals match.")
    else:
        logger.warning("✗ mismatch — re-run to resume (rsync) or investigate.")


# ── orchestration ─────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Copy /workspace from one RunPod network volume to another (two pods).",
    )
    p.add_argument("--source-volume-id", required=True)
    p.add_argument("--dest-volume-id", help="Existing destination volume id.")
    p.add_argument("--create-dest", action="store_true",
                   help="Create the destination volume (needs --dest-datacenter and --size).")
    p.add_argument("--dest-name", default="volcopy-dest", help="Name for a created dest volume.")
    p.add_argument("--size", type=int, help="Size (GB) for a created dest volume.")
    p.add_argument("--source-datacenter", help="Override source volume datacenter lookup.")
    p.add_argument("--dest-datacenter", help="Destination datacenter (required with --create-dest).")
    p.add_argument("--method", choices=["rsync", "send"], default="rsync")
    p.add_argument("--gpu-id", default=DEFAULT_GPU, help=f"GPU for transfer pods (default: {DEFAULT_GPU!r}).")
    p.add_argument("--compute-type", default="GPU", choices=["GPU", "CPU"])
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--ssh-key", default=DEFAULT_SSH_KEY, help="Local private key for pod SSH.")
    p.add_argument("--source-ssh", help="Manual source SSH, e.g. 'root@IP -p PORT'.")
    p.add_argument("--dest-ssh", help="Manual dest SSH, e.g. 'root@IP -p PORT'.")
    p.add_argument("--terminate-after", default="6h",
                   help="Safety TTL passed to pod create so pods self-terminate (default: 6h).")
    p.add_argument("--transfer-timeout", type=int, default=86400,
                   help="Seconds allowed for the transfer step (default: 24h).")
    p.add_argument("--keep-pods", action="store_true", help="Do not delete pods afterward.")
    p.add_argument("--dry-run", action="store_true", help="Resolve config and print the plan only.")
    return p


def _validate_args(args: argparse.Namespace) -> None:
    if not args.dest_volume_id and not args.create_dest:
        raise SystemExit("Provide --dest-volume-id, or --create-dest with --dest-datacenter/--size.")
    if args.create_dest and not (args.dest_datacenter and args.size):
        raise SystemExit("--create-dest requires --dest-datacenter and --size.")


def _run_copy(args: argparse.Namespace, src_dc: str, dst_dc: str, dest_volume_id: str) -> None:
    src_pod = dst_pod = None
    try:
        src_pod = create_pod("volcopy-src", args.source_volume_id, src_dc, args)
        dst_pod = create_pod("volcopy-dst", dest_volume_id, dst_dc, args)

        src_info = wait_pod_running(src_pod, POD_RUNNING_TIMEOUT)
        dst_info = wait_pod_running(dst_pod, POD_RUNNING_TIMEOUT)

        src_t = resolve_ssh_target(src_pod, src_info, args.ssh_key, args.source_ssh)
        dst_t = resolve_ssh_target(dst_pod, dst_info, args.ssh_key, args.dest_ssh)
        wait_ssh(src_t, SSH_READY_TIMEOUT)
        wait_ssh(dst_t, SSH_READY_TIMEOUT)

        if args.method == "rsync":
            transfer_rsync(src_t, dst_t, dst_info, args.transfer_timeout)
        else:
            transfer_send(src_t, dst_t, args.transfer_timeout)
        verify(src_t, dst_t)
    finally:
        if args.keep_pods:
            logger.info("--keep-pods: leaving %s and %s running.", src_pod, dst_pod)
        else:
            for pod in (src_pod, dst_pod):
                if pod:
                    delete_pod(pod)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _validate_args(args)

    src_dc = resolve_volume_dc(args.source_volume_id, args.source_datacenter)
    dst_dc = resolve_volume_dc(args.dest_volume_id, args.dest_datacenter) \
        if args.dest_volume_id else args.dest_datacenter

    logger.info("Plan: copy %s (%s) → %s (%s) via %s",
                args.source_volume_id, src_dc,
                args.dest_volume_id or f"[new '{args.dest_name}' {args.size}GB]", dst_dc,
                args.method)
    if args.dry_run:
        logger.info("--dry-run: stopping before creating anything.")
        return

    dest_volume_id = args.dest_volume_id or create_volume(args.dest_name, args.size, dst_dc)
    _run_copy(args, src_dc, dst_dc, dest_volume_id)
    logger.info("Done. Destination volume: %s", dest_volume_id)


if __name__ == "__main__":
    main()
