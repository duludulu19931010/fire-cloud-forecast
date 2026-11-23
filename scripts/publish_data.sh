#!/usr/bin/env bash
# 提交並推送 data 分支;推送衝突時 rebase 重試(預報與標註 workflow 共用同一分支)。
set -euo pipefail
cd data
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
if git diff --cached --quiet; then echo "nothing to publish"; exit 0; fi
git commit -m "${1:-update} $(date -u +%FT%TZ)"
for i in 1 2 3; do
  git push origin HEAD:data && exit 0
  git pull --rebase origin data || true
done
exit 1
