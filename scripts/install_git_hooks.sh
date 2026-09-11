#!/bin/sh
# Install this repository's git hooks: the staged-blob privacy scan (pre-commit) and the
# commit-message scan (commit-msg), both thin wrappers around scripts/git_precommit_scan.py.
#
# Hooks are machine-local by design (.git/hooks is never tracked), so `make setup` runs this
# and a fresh clone gets the gate the first time it builds its environment. Installed into the
# COMMON git dir, so commits made from linked agent worktrees run the same hooks.
#
# Refuses to overwrite a hook it did not install itself: an unrecognized hook is somebody
# else's machinery, and silently replacing it would disarm whatever it was checking.
set -eu

MARKER="installed by scripts/install_git_hooks.sh"
hooks_dir="$(git rev-parse --path-format=absolute --git-common-dir)/hooks"
mkdir -p "$hooks_dir"

install_hook() {
  name="$1"
  scan_args="$2"
  target="$hooks_dir/$name"
  if [ -e "$target" ] && ! grep -qF "$MARKER" "$target"; then
    echo "install-git-hooks: $target exists and is not ours; refusing to overwrite it." >&2
    echo "install-git-hooks: merge it with the wrapper this script writes, by hand." >&2
    exit 1
  fi
  cat >"$target" <<WRAPPER
#!/bin/sh
# $name privacy scan -- $MARKER; edit that script, not this wrapper.
top="\$(git rev-parse --show-toplevel)" || exit 1
script="\$top/scripts/git_precommit_scan.py"
if [ ! -f "\$script" ]; then
    # A linked-worktree checkout can predate the gate. The scan must still happen, so fall
    # back to the MAIN worktree's scanner (parent of the common git dir); the scanner already
    # arms from there, so which copy runs makes no difference to what gets scanned.
    common_dir="\$(git rev-parse --path-format=absolute --git-common-dir)" || exit 1
    top="\$(dirname "\$common_dir")"
    script="\$top/scripts/git_precommit_scan.py"
    if [ ! -f "\$script" ]; then
        echo "git-privacy-hook: scripts/git_precommit_scan.py missing from this checkout AND from the main worktree (\$top); REFUSING the commit rather than skipping the scan" >&2
        exit 1
    fi
fi
py="\$top/.venv/bin/python"
if [ ! -x "\$py" ]; then
    py="\$(command -v python3 || true)"
fi
if [ -z "\$py" ]; then
    echo "git-privacy-hook: no python3 on PATH and no .venv; refusing the commit rather than skipping the scan" >&2
    exit 1
fi
exec "\$py" "\$script" $scan_args
WRAPPER
  chmod +x "$target"
  echo "install-git-hooks: installed $target"
}

install_hook pre-commit ""
# The single quotes are the point: $1 must reach the wrapper verbatim, because git supplies the
# message-file path when it runs the commit-msg hook, not this installer.
# shellcheck disable=SC2016
install_hook commit-msg '--message-file "$1"'
