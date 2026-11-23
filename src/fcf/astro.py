"""太陽 / 月亮位置與事件時間(NOAA 演算法,精度約 ±1 分鐘,純標準庫)。"""
import math
from datetime import datetime, timezone

UTC = timezone.utc


def _jd(ts):
    return ts / 86400.0 + 2440587.5


def sun_pos(ts, lat, lon):
    """回傳 (仰角°, 方位角°[北=0,順時針]);未含大氣折射。"""
    T = (_jd(ts) - 2451545.0) / 36525.0
    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = math.radians(M)
    C = ((1.914602 - T * (0.004817 + 0.000014 * T)) * math.sin(Mr)
         + (0.019993 - 0.000101 * T) * math.sin(2 * Mr) + 0.000289 * math.sin(3 * Mr))
    om = 125.04 - 1934.136 * T
    lam = L0 + C - 0.00569 - 0.00478 * math.sin(math.radians(om))
    eps = (23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
           + 0.00256 * math.cos(math.radians(om)))
    dec = math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(lam)))
    y = math.tan(math.radians(eps) / 2) ** 2
    L0r = math.radians(L0)
    eqt = 4 * math.degrees(
        y * math.sin(2 * L0r) - 2 * e * math.sin(Mr)
        + 4 * e * y * math.sin(Mr) * math.cos(2 * L0r)
        - 0.5 * y * y * math.sin(4 * L0r) - 1.25 * e * e * math.sin(2 * Mr))
    utc_min = (ts % 86400) / 60.0
    ha = math.radians(((utc_min + eqt + 4 * lon) % 1440) / 4 - 180)
    la = math.radians(lat)
    sin_el = math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)
    el = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    az = (math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(la)
                                  - math.tan(dec) * math.cos(la))) + 180) % 360
    return el, az


def crossings(day_start_ts, lat, lon, target_el, step=600):
    """24 小時內太陽仰角穿越 target_el 的時刻 → [(ts, 'rise'|'set')]。"""
    out = []
    pt, pe = day_start_ts, sun_pos(day_start_ts, lat, lon)[0] - target_el
    for k in range(1, 24 * 3600 // step + 1):
        t = day_start_ts + k * step
        e = sun_pos(t, lat, lon)[0] - target_el
        if pe * e < 0:
            a, b, fa = pt, t, pe
            for _ in range(30):
                m = (a + b) / 2
                fm = sun_pos(m, lat, lon)[0] - target_el
                if fa * fm <= 0:
                    b = m
                else:
                    a, fa = m, fm
            out.append(((a + b) / 2, "rise" if e > pe else "set"))
        pt, pe = t, e
    return out


def day_events(d, lat, lon, tz_h=8):
    """當地日期 d 的日出/日落(-0.833°)與民用/航海曙暮光(俯角 6°、12°)。"""
    t0 = datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() - tz_h * 3600
    out = {"date": d.isoformat()}
    for c, kind in crossings(t0, lat, lon, -0.833):
        out["sunrise" if kind == "rise" else "sunset"] = c
    for dep in (6, 12):
        for c, kind in crossings(t0, lat, lon, -float(dep)):
            out[("dawn" if kind == "rise" else "dusk") + str(dep)] = c
    return out


def moon_illum(ts):
    """月面照亮比例 0~1(平均朔望月近似,誤差約 ±5%)。"""
    phase = ((_jd(ts) - 2451550.1) / 29.530588853) % 1.0
    return (1 - math.cos(2 * math.pi * phase)) / 2
