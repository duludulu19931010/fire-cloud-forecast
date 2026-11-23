"""球面幾何與「光路」物理:陽光要穿過多遠、多低的大氣才能照到雲底。"""
import math

R = 6371.0088                                   # km
H = {"low": 1.2, "mid": 4.5, "high": 9.0}       # 各雲層代表高度 km(低:<2、中:2-6、高:>6)
KAPPA = {"low": 0.95, "mid": 0.75, "high": 0.35}  # 該層雲量對直射光的衰減係數
LCORR = 150.0                                   # 雲場空間相關長度 km(避免把取樣點當獨立事件)


def dest(lat, lon, az_deg, km):
    d, a = km / R, math.radians(az_deg)
    la, lo = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(a))
    lo2 = lo + math.atan2(math.sin(a) * math.sin(d) * math.cos(la),
                          math.cos(d) - math.sin(la) * math.sin(la2))
    return math.degrees(la2), (math.degrees(lo2) + 540) % 360 - 180


def snap(v, step=0.25):
    """對齊 0.25° 格點:模式解析度本來就是這個量級,可讓不同地點共用取樣點、省 API 額度。"""
    return round(round(v / step) * step, 4)


def lit_min_height(delta_deg, x_km):
    """太陽在觀測者地平線下 delta°時,距離 x_km 處、高度至少多少 km 的雲才在地球陰影之外。"""
    dp = math.radians(delta_deg) - x_km / R
    return R * dp * dp / 2 if dp > 0 else 0.0


def ray_alt(h, x, delta_deg, u):
    """照亮 (距離 x, 高度 h) 的那道光,在更靠太陽方向 u 處離地多高(km)。"""
    dp = math.radians(delta_deg) - x / R
    s = u - x
    return h - s * math.tan(dp) + s * s / (2 * R)


def elev_angle(h, u):
    """高度 h、水平距離 u 的雲,在觀測者眼中的仰角(°),含地球曲率。"""
    if u <= 0:
        return 90.0
    return math.degrees(math.atan2(h - u * u / (2 * R), u))


def path_T(h, x, delta, us, covers):
    """光路穿透率 0~1。us=沿太陽方位的取樣距離;covers[i][layer]=該點該層雲量 0~1。"""
    if lit_min_height(delta, x) >= h:
        return 0.0                                # 雲在地球陰影裡
    T, n = 1.0, len(us)
    for i, u in enumerate(us):
        if u <= x:
            continue
        a = ray_alt(h, x, delta, u)
        if a <= 0.05:
            return 0.0
        lay = "low" if a < 2 else "mid" if a < 6 else "high"
        lo = (us[i - 1] + u) / 2 if i > 0 else u
        hi = (u + us[i + 1]) / 2 if i < n - 1 else u + (u - us[i - 1]) / 2
        T *= (1 - KAPPA[lay] * covers[i][lay]) ** (max(hi - lo, 0.0) / LCORR)
    return T


def graze_alt(h, x, delta):
    """照到 (x, h) 的光線,沿途離地最低點(km)。越低 = 穿過越厚的大氣 = 越紅。"""
    dp = math.radians(delta) - x / R
    return h - R * dp * dp / 2 if dp > 0 else h


def redness(alt_min):
    """泛紅係數 0~1:掠過高度 ≥8 km 近乎直射(偏白/金)→0;2 km 以下最紅;<1 km 光線衰弱漸暗。"""
    red = max(0.0, min(1.0, (8.0 - alt_min) / 6.5))
    return red * (0.6 + 0.4 * max(0.0, min(1.0, alt_min / 1.0)))
