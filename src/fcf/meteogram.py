"""Meteogram(氣象圖)SVG:同一條時間軸上疊「高/中/低雲層」「事件機率與高峰窗口」「山區雲海/琉璃光逐時分數」。
概念參考 meteoblue / ECMWF 的 meteogram:一張圖看懂「雲在哪一層、什麼時候、值不值得去」。純標準庫,輸出 charts/<site>.svg。"""
from datetime import datetime, timedelta, timezone
from html import escape

from .scoring import interp

W, H, L, R = 760, 330, 56, 14
KIND_ZH = {"sunset": "日落", "sunrise": "日出", "sea_of_clouds": "雲海", "ryuli": "琉璃光"}


def _x(ts, t0, t1):
    return L + (ts - t0) / (t1 - t0) * (W - L - R)


def cloud_layers(site_pt, weights, now, hours=24):
    out = []
    for h in range(hours + 1):
        ts = int(now // 3600 * 3600 + h * 3600)
        row = {}
        for lay in ("high", "mid", "low"):
            vs = [v for m in weights if (v := interp(site_pt, m, f"cloud_cover_{lay}", ts)) is not None]
            row[lay] = sum(vs) / len(vs) if vs else None
        out.append((ts, row))
    return out


def render(site, layers, events, timeline, now, tz=8, hours=24):
    t0, t1 = int(now // 3600 * 3600), int(now // 3600 * 3600) + hours * 3600
    tzinfo = timezone(timedelta(hours=tz))
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">',
         '<style>:root{--fg:#222;--bg:#fff;--grid:#ddd}@media(prefers-color-scheme:dark){:root{--fg:#ddd;--bg:#1b1b1b;--grid:#444}}'
         'text{fill:var(--fg)}</style>', f'<rect width="{W}" height="{H}" fill="var(--bg)"/>',
         f'<text x="{L}" y="16" font-size="13" font-weight="600">{escape(site["name"])}・未來 {hours} 小時</text>']
    for h in range(0, hours + 1, 3):                                        # 時間軸
        ts = t0 + h * 3600
        x = _x(ts, t0, t1)
        s.append(f'<line x1="{x:.1f}" y1="26" x2="{x:.1f}" y2="{H - 24}" stroke="var(--grid)"/>')
        s.append(f'<text x="{x:.1f}" y="{H - 8}" text-anchor="middle">{datetime.fromtimestamp(ts, tzinfo):%H:%M}</text>')
    y = 34                                                                  # 雲層熱度條(高→低)
    for lay, name in (("high", "高雲"), ("mid", "中雲"), ("low", "低雲")):
        s.append(f'<text x="{L - 6}" y="{y + 12}" text-anchor="end">{name}</text>')
        for (ts, row), (ts2, _) in zip(layers, layers[1:]):
            v = row[lay]
            if v is None:
                continue
            x1, x2 = _x(ts, t0, t1), _x(ts2, t0, t1)
            s.append(f'<rect x="{x1:.1f}" y="{y}" width="{x2 - x1 + .5:.1f}" height="18" fill="#4a7fb5" opacity="{0.08 + 0.9 * v / 100:.2f}"/>')
        y += 22
    top = y + 8                                                             # 事件窗口 + 機率
    s.append(f'<text x="{L - 6}" y="{top + 30}" text-anchor="end">機率</text>')
    for e in events:
        a, b = _x(max(e["window_ts"][0], t0), t0, t1), _x(min(e["window_ts"][1] + 600, t1), t0, t1)
        if b < L or a > W - R:
            continue
        hgt = 60 * e["prob"]
        col = "#e0662b" if e["kind"] in ("sunset", "sunrise") else "#6a5acd"
        s.append(f'<rect x="{a:.1f}" y="{top + 60 - hgt:.1f}" width="{max(b - a, 6):.1f}" height="{hgt:.1f}" fill="{col}" opacity=".75"/>')
        s.append(f'<text x="{(a + b) / 2:.1f}" y="{top + 56 - hgt:.1f}" text-anchor="middle">{KIND_ZH[e["kind"]]} {e["prob"] * 100:.0f}%</text>')
    base = top + 60
    s.append(f'<line x1="{L}" y1="{base}" x2="{W - R}" y2="{base}" stroke="var(--grid)"/>')
    if timeline:                                                            # 山區:逐時雲海 / 琉璃光分數
        y0 = base + 22
        for key, col, name, dy in (("sea", "#2a9d8f", "雲海", 0), ("ryuli", "#6a5acd", "琉璃光", 1)):
            s.append(f'<text x="{L - 6}" y="{y0 + 44 * dy + 28}" text-anchor="end">{name}</text>')
            pts = [(_x(t["ts"], t0, t1), y0 + 44 * dy + 40 - 40 * t[key] / 100) for t in timeline if t0 <= t["ts"] <= t1]
            if len(pts) > 1:
                s.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (col, " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)))
    s.append("</svg>")
    return "\n".join(s)
