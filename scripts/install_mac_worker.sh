#!/usr/bin/env bash
# Install the Loom worker on an Apple Silicon Mac and keep it running.
#
# Not a container, and not for want of trying: Metal is a macOS userspace
# framework, and a Linux container on a Mac is a guest in a VM with no GPU
# device at all — not even /dev/dri. There is no passthrough to enable, so the
# GPU is reachable only from a native process.
#
# What this gives instead is the same two properties `docker run --restart
# unless-stopped` gives: one command to set up, and a service that comes back
# by itself after a crash or a reboot. That is launchd's job on macOS.
#
# From a checkout:
#   bash scripts/install_mac_worker.sh --key loom_<...>
#
# On a machine that has never seen this repository — one line, nothing to
# clone, no git required (pip fetches the source archive itself):
#   curl -fsSL https://raw.githubusercontent.com/gihpee/loom/main/scripts/install_mac_worker.sh | bash -s -- --key loom_<...>
#
# Uninstall:
#   launchctl bootout gui/$(id -u)/network.loom.worker
#   rm ~/Library/LaunchAgents/network.loom.worker.plist
set -euo pipefail

LABEL="network.loom.worker"
PREFIX="${LOOM_PREFIX:-$HOME/.loom}"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
KEY=""
ORCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --key) KEY="$2"; shift 2 ;;
    --orchestrator) ORCH="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0" 2>/dev/null || echo "usage: --key loom_<...>"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$KEY" ]]; then
  echo "usage: $0 --key loom_<join key>  [--orchestrator host:port]" >&2
  exit 2
fi

# --- the machine ------------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This installer is for Apple Silicon. On other hardware use the" >&2
  echo "container image: docker run --gpus all gihpee/loomworker --key ..." >&2
  exit 1
fi

# Python 3.11+ that is genuinely arm64. Under Rosetta the MLX wheels install
# but there is no Metal behind them, which fails much later and confusingly.
PY="${LOOM_PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "python3 3.11+ required (found $("$PY" --version 2>&1))" >&2
  exit 1
fi
if [[ "$("$PY" -c 'import platform; print(platform.machine())')" != "arm64" ]]; then
  echo "this python is not native arm64 — MLX would install without a GPU" >&2
  exit 1
fi

# --- where the worker comes from --------------------------------------------
# Run from a checkout, it installs that checkout — the developer's case, and
# the one where installing something else would be baffling. Piped from curl
# there is no checkout at all (BASH_SOURCE is the pipe), so pip fetches the
# repository archive instead.
#
# An archive URL rather than git+https on purpose: git on a fresh Mac means
# the Xcode command line tools, which is a multi-gigabyte download and a
# dialog box. pip needs neither to unpack a tarball.
SOURCE="${LOOM_SOURCE:-}"
if [[ -z "$SOURCE" ]]; then
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
  if [[ -n "$HERE" && -f "$HERE/../worker/pyproject.toml" ]]; then
    SOURCE="$(cd "$HERE/.." && pwd)/worker"
    echo "installing from this checkout: $SOURCE"
  else
    BRANCH="${LOOM_BRANCH:-main}"
    SOURCE="https://github.com/gihpee/loom/archive/refs/heads/$BRANCH.tar.gz#subdirectory=worker"
    echo "installing from github ($BRANCH)"
  fi
fi

# --- install ----------------------------------------------------------------
echo "installing into $PREFIX"
"$PY" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install --quiet --upgrade pip
# [p2p] as well as [mlx]: without it every activation is relayed through
# the orchestrator, which on a spread-out pipeline is the whole cost.
if [[ "$SOURCE" == http* ]]; then
  "$PREFIX/venv/bin/pip" install --quiet --upgrade "loom-worker[mlx,p2p] @ $SOURCE"
else
  "$PREFIX/venv/bin/pip" install --quiet --upgrade "$SOURCE[mlx,p2p]"
fi

echo -n "checking the GPU is actually reachable... "
"$PREFIX/venv/bin/python" - <<'PY'
import sys
import mlx.core as mx
device = mx.default_device()
a = mx.random.normal((256, 256)); mx.eval(a @ a)
if "gpu" not in str(device).lower():
    print(f"FAILED: MLX chose {device}, not the GPU", file=sys.stderr)
    raise SystemExit(1)
print(device)
PY

# --- run it as a service ----------------------------------------------------
mkdir -p "$HOME/Library/LaunchAgents" "$PREFIX/logs"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PREFIX/venv/bin/loom-worker</string>
    <string>--key</string><string>$KEY</string>
$( [[ -n "$ORCH" ]] && printf '    <string>--orchestrator</string><string>%s</string>\n' "$ORCH" )
  </array>
  <!-- Start at login and restart after a crash: the same promise as
       docker run --restart unless-stopped. -->
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$PREFIX/logs/worker.log</string>
  <key>StandardErrorPath</key><string>$PREFIX/logs/worker.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HF_HOME</key><string>$PREFIX/huggingface</string>
    <key>LOOM_P2P_KEY_DIR</key><string>$PREFIX/p2p</string>
  </dict>
</dict>
</plist>
PLIST

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo
echo "worker installed and started."
echo "  logs:    tail -f $PREFIX/logs/worker.log"
echo "  stop:    launchctl bootout gui/$(id -u)/$LABEL"
echo "  start:   launchctl bootstrap gui/$(id -u) $PLIST"
echo
echo "  update:  re-run this same command"
echo
echo "Deploy a model to it from the orchestrator's Deploy tab with backend"
echo "'mlx_shard' — this node can serve a range of layers like any other."
