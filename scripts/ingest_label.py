"""把 GitHub Issue 表單(觀測回報)寫進 data/labels.csv。環境變數: ISSUE_BODY。
成功印出摘要(exit 0);輸入不合法 exit 1 並印出原因(workflow 會回覆在 issue 裡)。"""
import csv
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone

KINDS = {"sunset", "sunrise", "sea_of_clouds", "ryuli"}


def parse(body):
    return {m.group(1).strip(): m.group(2).strip()
            for m in re.finditer(r"###\s+(.+?)\n+(.*?)(?=\n###\s|\Z)", body + "\n", re.S)}


def main():
    cfg = json.load(open("config/sites.json", encoding="utf-8"))
    ids = {s["id"] for s in cfg["sites"]}
    f = parse(os.environ.get("ISSUE_BODY", ""))
    try:
        site = f["地點"].split()[0]
        kind = f["類型"].split()[0]
        d = date.fromisoformat(f["日期"].split()[0])
        label = int(f["評級"].split()[0])
    except (KeyError, ValueError, IndexError) as e:
        print(f"欄位解析失敗:{e!r}")
        return 1
    today = (datetime.now(timezone.utc) + timedelta(hours=cfg.get("tz_offset", 8))).date()
    if site not in ids or kind not in KINDS or not 0 <= label <= 3 or d > today:
        print(f"內容不合法:site={site} kind={kind} date={d} label={label}")
        return 1
    path = "data/labels.csv"
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "site", "kind", "label", "source", "note"])
        w.writerow([d.isoformat(), site, kind, label, "issue", ("" if (f.get("備註") or "").startswith("_No response_") else (f.get("備註") or "")).replace("\n", " ")[:200]])
    print(f"已記錄:{d} {site} {kind} 評級 {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
