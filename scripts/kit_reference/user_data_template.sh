#!/bin/bash
# Reference user-data for a rented GPU box. Rendered by scripts/render_box_userdata.sh through
# scripts/launch_gpu_box.sh; scripts/kit_reference/README.md documents every placeholder and which
# ones the launcher fills on its own. Order is the killswitch contract: dead-man first, the ship guard
# verifies the code by content before anything runs it, boot patch before the job, watchdog as a root
# unit outside tmux, runner last. Never name a placeholder inside a comment: the render refuses it.
exec >>/var/log/box-bootstrap.log 2>&1
set -ux
shutdown -h +@DEADMAN_MINUTES@ "@RUN_NAME@: max-lifetime dead-man"

# The guard body inlined below exports and consumes these; shellcheck sees only the declarations.
SHIP_TREE=@SHIP_TREE@
# shellcheck disable=SC2034
SHIP_CODE_SHA256=@SHIP_CODE_SHA256@
# shellcheck disable=SC2034
SHIP_TREE_DIGEST=@SHIP_TREE_DIGEST@
# shellcheck disable=SC2034
SHIP_S3_KEY=@SHIP_S3_KEY@
SHIP_S3_REGION=@SHIP_S3_REGION@
SHIP_EXTRACT_DIR=/home/ubuntu/repo
@SHIP_GUARD@

chown -R ubuntu:ubuntu "$SHIP_EXTRACT_DIR"
echo "@SHIP_HEAD_SHA@" >"$SHIP_EXTRACT_DIR/GIT_SHA"
echo "$SHIP_TREE" >"$SHIP_EXTRACT_DIR/SHIP_TREE"

TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')
IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)
echo "$IID" >/home/ubuntu/IID
echo "instance $IID run @RUN_NAME@ tree $SHIP_TREE"
echo "bootstrapping started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >/home/ubuntu/CHAIN_STATE
chown ubuntu:ubuntu /home/ubuntu/IID /home/ubuntu/CHAIN_STATE

bash "$SHIP_EXTRACT_DIR/scripts/ec2_boot_patch.sh" \
  || echo "WARN: boot patch failed, see /var/log/boot-patch.log"

systemd-run --unit=box-idle-watchdog --collect /bin/bash -c \
  "IDLE_WATCHDOG_S3_DEST=@IDLE_WATCHDOG_S3_DEST@ $SHIP_EXTRACT_DIR/scripts/idle_watchdog.sh --minutes @IDLE_MINUTES@ --done-marker /home/ubuntu/AGENDA_DONE >>/var/log/idle-watchdog.log 2>&1"

RUNNER_TREE_PATH=scripts/kit_reference/runner_skeleton.sh
RUN_NAME="@RUN_NAME@" RUN_PREFIX="@RUN_PREFIX@" S3_BASE="@SHIP_S3_PREFIX@" S3_REGION="$SHIP_S3_REGION" \
  ARM="@ARM@" MODEL="@MODEL@" MAX_STEPS="@MAX_STEPS@" EVAL_STEPS="@EVAL_STEPS@" \
  PLAN_STEP_MINUTES="@PLAN_STEP_MINUTES@" DONE_MARKER=/home/ubuntu/AGENDA_DONE \
  REPO="$SHIP_EXTRACT_DIR" bash "$SHIP_EXTRACT_DIR/$RUNNER_TREE_PATH"
