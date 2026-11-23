#!/usr/bin/env bash
# 把 data 分支掛成 ./data(不存在就建立孤兒分支)。data 分支放預報 JSON、歷史、驗證、模型參數、labels.csv。
set -euo pipefail
git config --global --add safe.directory "$PWD"
if git ls-remote --exit-code --heads origin data >/dev/null 2>&1; then
  git fetch --no-tags origin data
  git worktree add -B data data origin/data
else
  git worktree add --detach data
  git -C data checkout --orphan data
  git -C data rm -rf . >/dev/null 2>&1 || true
fi
