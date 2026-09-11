#!/usr/bin/env bash
# OS security patching at instance boot. Run as root from user-data, after the dead-man switch is
# armed and before the job starts. Also safe to run on a live box over SSM (it holds the GPU stack,
# so nothing a running job depends on gets swapped underneath it).
#
# Why this exists: a box carries whatever known-critical CVEs its image shipped with from the moment
# it boots, however few hours it then lives, and no killswitch sizing shortens that window -- only
# patching closes it. Nor does "always launch the latest AMI": the critical that earned this pass
# (2026-08-20) shipped in the newest available DLAMI build, and its fix lived in the -updates pocket
# rather than -security, so a security-pocket-only pass (stock unattended-upgrades) would have missed
# it. Hence: full upgrade, with the driver stack held.
#
# The NVIDIA/CUDA hold is load-bearing. Upgrading the driver userspace under a loaded kernel
# module breaks CUDA ("driver/library version mismatch") until reboot, and these boxes exist to
# run one GPU job and die. Holds are deliberately left in place afterwards: a later manual apt run
# on the box should not silently bump the driver either.
#
# What this pass does NOT do, deliberately: bump the kernel. The base GPU image ships its own
# apt-holds on the kernel metapackages (linux-aws, linux-image-aws, linux-headers-aws,
# linux-modules-extra-aws) and the lustre modules built against that ABI, and `upgrade
# --with-new-pkgs` does not defeat a hold -- only an unhold would, and the log line below reports
# those six as still-upgradable on every run, which is expected rather than a failure. Refreshing
# the kernel belongs at image-bake time, where nothing is running, the box boots fresh into the new
# kernel, and an image build can verify that the newest kernel has a matching NVIDIA module before
# publishing. On a live box the same unhold would install a kernel nothing has verified a driver
# against, which risks the GPU stack for no gain this pass can deliver.
#
# Deliberately not `set -e`: a mirror hiccup must not kill the science job the box exists for.
# Failure is loud instead -- the status file and log line below are what monitoring reads.
set -u

LOG=/var/log/boot-patch.log
STATUS_FILE=/var/run/boot-patch-status
# Always-Include-Phased-Updates: Ubuntu phases fixed packages to a random cohort over days, and a
# box outside the cohort would keep the vulnerable version through a "successful" full upgrade.
APT_OPTS="-o DPkg::Lock::Timeout=600 -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold -o APT::Get::Always-Include-Phased-Updates=true"
export DEBIAN_FRONTEND=noninteractive

log() {
  echo "[boot-patch $(date -u +%FT%TZ)] $*" | tee -a "$LOG"
}

apt_retry() {
  # Two attempts with a pause: transient mirror/lock failures are common in the first boot minute.
  local attempt
  for attempt in 1 2; do
    # shellcheck disable=SC2086  # APT_OPTS is a deliberate word-split of single-token options
    if apt-get $APT_OPTS "$@" >>"$LOG" 2>&1; then
      return 0
    fi
    log "apt-get $* failed (attempt $attempt)"
    sleep 20
  done
  return 1
}

log "starting OS security patch pass"

# Pin the GPU stack and the container runtime before anything else so no upgrade path can touch
# them. The runtime hold exists because this same body also runs on live boxes over SSM: swapping
# docker/containerd under a running container kills the job it hosts, the same failure mode as the
# driver-userspace swap the header describes.
dpkg-query -f '${Package}\n' -W 'nvidia-*' 'libnvidia-*' 'cuda-*' \
  'docker.io' 'docker-ce*' 'containerd*' 'runc' 2>/dev/null | xargs -r apt-mark hold >>"$LOG" 2>&1 || true

rc=0
apt_retry update || rc=1
if [ "$rc" -eq 0 ]; then
  # --with-new-pkgs so kernel meta-package bumps (which pull in a new linux-image-*) are not
  # silently "kept back" -- those are exactly the packages a CVE scan flags.
  apt_retry upgrade -y --with-new-pkgs || rc=1
fi

remaining=$(apt list --upgradable 2>/dev/null | grep -c upgradable || true)
if [ "$rc" -eq 0 ]; then
  echo "OK" >"$STATUS_FILE"
  log "patch pass complete; $remaining packages still upgradable (held GPU stack expected among them)"
else
  echo "FAILED" >"$STATUS_FILE"
  log "PATCH PASS FAILED -- box is running with unpatched packages; expect CVE findings against it"
fi
exit "$rc"
