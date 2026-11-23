"""事件規劃:哪個地點、哪一類現象、什麼時間窗口。(預報、回測、真值共用)"""
from datetime import date, datetime, timedelta, timezone

from . import astro, geo, scoring

UTC = timezone.utc
GLOW_SPAN = {"sunset": (-15 * 60, 45 * 60), "sunrise": (-50 * 60, 8 * 60)}
KINDS = ("sunset", "sunrise", "sea_of_clouds", "ryuli")


def key(lat, lon):
    return (round(lat, 4), round(lon, 4))


def local_today(now, tz):
    return (datetime.fromtimestamp(now, UTC) + timedelta(hours=tz)).date()


def make_event(site, kind, day, ev, nxt):
    if kind in GLOW_SPAN:
        ref = ev.get(kind)
        if ref is None:
            return None
        a, b = GLOW_SPAN[kind]
        return {"site": site, "kind": kind, "date": day.isoformat(), "ref": ref,
                "az": astro.sun_pos(ref, site["lat"], site["lon"])[1], "t0": ref + a, "t1": ref + b}
    if kind == "sea_of_clouds" and ev.get("sunrise"):
        t0, t1 = ev["sunrise"] - 3600, ev["sunrise"] + 7200
    elif kind == "ryuli" and ev.get("dusk12") and nxt.get("dawn12"):
        t0, t1 = ev["dusk12"], nxt["dawn12"]
    else:
        return None
    return {"site": site, "kind": kind, "date": day.isoformat(), "t0": t0, "t1": t1}


def _event(site, kind, day, tz):
    return make_event(site, kind, day, astro.day_events(day, site["lat"], site["lon"], tz),
                      astro.day_events(day + timedelta(days=1), site["lat"], site["lon"], tz))


def plan(cfg, now, horizon=24):
    """與「現在起 horizon 小時」有交集的事件(含昨晚開始、尚未結束的琉璃光)。"""
    tz, today, out = cfg.get("tz_offset", 8), local_today(now, cfg.get("tz_offset", 8)), []
    for site in cfg["sites"]:
        for d in (-1, 0, 1):
            day = today + timedelta(days=d)
            ev = astro.day_events(day, site["lat"], site["lon"], tz)
            nxt = astro.day_events(day + timedelta(days=1), site["lat"], site["lon"], tz)
            for kind in site["types"]:
                e = make_event(site, kind, day, ev, nxt)
                if e and e["t1"] >= now and e["t0"] <= now + horizon * 3600:
                    out.append(e)
    return out


def plan_ended(cfg, keys, now, min_wait=3600, max_age=3 * 86400):
    """keys=(site_id, kind, date) 集合 → 已結束 min_wait 秒以上、未超過 max_age 的事件。"""
    tz, sites, out = cfg.get("tz_offset", 8), {s["id"]: s for s in cfg["sites"]}, []
    for sid, kind, ds in sorted(keys):
        s = sites.get(sid)
        if not s:
            continue
        e = _event(s, kind, date.fromisoformat(ds), tz)
        if e and now - max_age < e["t1"] <= now - min_wait:
            out.append(e)
    return out


def path_keys(site, az):
    return [key(*(geo.snap(v) for v in geo.dest(site["lat"], site["lon"], az, u))) for u in scoring.US[1:]]
