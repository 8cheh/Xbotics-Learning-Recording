#!/usr/bin/env bash
# Install the WSL-side backend shipped next to this file.
set -eu

TARGET="${HOME}/.local/bin/linux-gui"
SOURCE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SOURCE="${SOURCE_DIR}/linux-gui"

[[ -f "$SOURCE" ]] || { printf '[XX] missing %s\n' "$SOURCE" >&2; exit 1; }
mkdir -p "$(dirname "$TARGET")"
install -m 0755 "$SOURCE" "$TARGET"
printf '[OK] installed %s\n' "$TARGET"

if command -v apt-get >/dev/null 2>&1; then
  missing=()
  for command in dbus-run-session startxfce4 Xephyr; do
    command -v "$command" >/dev/null 2>&1 || missing+=("$command")
  done
  if ((${#missing[@]})); then
    printf '[!!] missing commands: %s\n' "${missing[*]}"
    printf '     install packages with: sudo apt-get update && sudo apt-get install xfce4 xfce4-goodies xfce4-terminal thunar xserver-xephyr dbus-x11\n'
  fi
fi

"$TARGET" --check || true
