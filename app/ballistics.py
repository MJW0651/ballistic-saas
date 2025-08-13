from dataclasses import dataclass
from math import radians, sin, cos
import json
import pathlib

from .schemas import SolveInput

# ----------------- Constants -----------------
G_FTPS2 = 32.174
IN_PER_FT = 12.0
FT_PER_YD = 3.0
FTPS_PER_MPH = 1.4666667

# Fallback (when canonical tables missing)
BASE_K = 3.5e-5   # increase for MORE drift, decrease for LESS (fallback only)
REF_C  = 1.0e-3

_DRAG_JSON = {}
_DRAG_TABLE = {}

@dataclass
class Row:
    yd: int
    elev: float
    wind: float     # signed: +R, -L
    tof_s: float
    vel_fps: float
    energy_ftlb: float

# ----------------- Helpers -----------------
def to_mil(inches: float, yards: float) -> float:
    return round((inches / (yards * 36.0)) * 1000.0, 2)

def to_moa(inches: float, yards: float) -> float:
    moa = inches / (1.0472 * (yards / 100.0))
    return round(moa, 1)

def energy_ftlb(vel_fps: float, bullet_gr: float = 140.0) -> float:
    return (bullet_gr * vel_fps**2) / 450240.0

def _air_density_scale(temp_f: float, pressure_inHg: float, rh: float, altitude_ft: float) -> float:
    T_R = temp_f + 459.67
    return (max(0.1, pressure_inHg) / 29.92) * (519.67 / max(200.0, T_R))

# ----------------- Drag loading -----------------
def _load_table_or_fallback(model: str):
    """
    Try canonical G-table mcg{1,7}.txt (velocity fps, f(V) per line).
    If missing, fall back to starter JSON and mark mode='fallback'.
    Returns: (callable f(v_fps), mode) where mode in {'canonical','fallback'}
    """
    key = (model or "G7").upper()
    here = pathlib.Path(__file__).parent
    drag_dir = here / "drag"
    table_name = {"G1": "mcg1.txt", "G7": "mcg7.txt"}.get(key, "mcg7.txt")
    table_path = drag_dir / table_name

    if table_path.exists():
        if key not in _DRAG_TABLE:
            xs, ys = [], []
            for line in table_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    v = float(parts[0]); g = float(parts[1])
                    xs.append(v); ys.append(g)
                except ValueError:
                    continue
            pairs = sorted(zip(xs, ys), key=lambda p: p[0], reverse=True)
            _DRAG_TABLE[key] = pairs
        pairs = _DRAG_TABLE[key]

        def f_of_v(v_fps: float) -> float:
            if v_fps >= pairs[0][0]:
                return pairs[0][1]
            if v_fps <= pairs[-1][0]:
                return pairs[-1][1]
            for (v1, g1), (v2, g2) in zip(pairs, pairs[1:]):
                if v2 <= v_fps <= v1:
                    t = (v1 - v_fps) / (v1 - v2)
                    return g1 + t * (g2 - g1)
            return pairs[-1][1]

        return f_of_v, "canonical"

    # ---- Fallback JSON ----
    key = "G7" if key not in ("G1", "G7") else key
    json_path = drag_dir / f"{key.lower()}.json"
    if key not in _DRAG_JSON:
        _DRAG_JSON[key] = json.loads(json_path.read_text())
    tbl = _DRAG_JSON[key]

    def f_of_v_fallback(v_fps: float) -> float:
        if v_fps >= tbl[0][0]:
            val = tbl[0][1]
        elif v_fps <= tbl[-1][0]:
            val = tbl[-1][1]
        else:
            for (v1, d1), (v2, d2) in zip(tbl, tbl[1:]):
                if v2 <= v_fps <= v1:
                    t = (v1 - v_fps) / (v1 - v2)
                    val = d1 + t * (d2 - d1)
                    break
        return float(val)

    return f_of_v_fallback, "fallback"

# ----------------- Solver -----------------
def compute_table(inp: SolveInput):
    """
    Integrator with:
      - Canonical Siacci/point-mass when canonical tables present:
          a_drag = - (g * f(V_rel) * rho_ratio / BC) * v_rel_hat
      - Calibrated quadratic fallback when tables missing:
          a_drag = - k * |v_rel| * v_rel  with k tuned from JSON scale
    Wind 'coming from' convention:
      0° headwind, 90° from right (R->L), 180° tailwind, 270° from left (L->R).
    """
    f_of_v, mode = _load_table_or_fallback(inp.bc_model)

    v0 = float(inp.mv_fps)
    zero_ft = inp.zero_yd * FT_PER_YD
    scope_h_ft = inp.scope_h_in / IN_PER_FT
    rho_ratio = _air_density_scale(inp.temp_f, inp.pressure_inHg, inp.rh, inp.altitude_ft)

    # Seed muzzle angle from no-drag zero
    theta = ((G_FTPS2 * zero_ft**2) / (2.0 * v0**2) - scope_h_ft) / max(1.0, zero_ft)

    # Initial state (ground-relative)
    vx, vy, vz = v0, v0 * theta, 0.0
    x = y = z = 0.0
    t = 0.0

    # Distances
    dists = list(range(inp.start_yd, inp.end_yd + 1, inp.step_yd))
    next_i = 0

    # Wind vector (ft/s), where wind_dir_deg is "coming from"
    wind_speed = max(0.0, float(inp.wind_speed_mph)) * FTPS_PER_MPH
    ang = radians(float(inp.wind_dir_deg))
    wind_x = -wind_speed * cos(ang)  # 0° -> -x
    wind_z = -wind_speed * sin(ang)  # 90° -> -z (from right)

    dt = 0.001 if getattr(inp, "solver_quality", "standard") == "high" else 0.002
    bullet_gr = float(getattr(inp, "bullet_gr", 140.0))
    rows = []

    while next_i < len(dists) and t < 6.0 and vx > 0.1:
        # Air-relative velocity
        rx = vx - wind_x
        ry = vy
        rz = vz - wind_z
        vrel = (rx*rx + ry*ry + rz*rz)**0.5 or 1e-6

        if mode == "canonical":
            # Siacci: a = -g * f * rho_ratio / BC * v_rel_hat
            fval = f_of_v(vrel)
            k = (G_FTPS2 * fval * rho_ratio) / max(1e-6, inp.bc)
            ax = -k * (rx / vrel)
            ay = -G_FTPS2 - k * (ry / vrel)
            az = -k * (rz / vrel)
        else:
            # Fallback (calibrated quadratic)
            scale = f_of_v(vrel)  # ~1e-3 .. 1e-2 from our JSON
            k = (BASE_K * (scale / REF_C) * rho_ratio) / max(1e-6, inp.bc)
            ax = -k * rx * vrel
            ay = -G_FTPS2 - k * ry * vrel
            az = -k * rz * vrel

        # RK2 midpoint
        mx = vx + ax * dt * 0.5
        my = vy + ay * dt * 0.5
        mz = vz + az * dt * 0.5

        mrx = mx - wind_x
        mry = my
        mrz = mz - wind_z
        vrel2 = (mrx*mrx + mry*mry + mrz*mrz)**0.5 or 1e-6

        if mode == "canonical":
            fval2 = f_of_v(vrel2)
            k2 = (G_FTPS2 * fval2 * rho_ratio) / max(1e-6, inp.bc)
            ax2 = -k2 * (mrx / vrel2)
            ay2 = -G_FTPS2 - k2 * (mry / vrel2)
            az2 = -k2 * (mrz / vrel2)
        else:
            scale2 = f_of_v(vrel2)
            k2 = (BASE_K * (scale2 / REF_C) * rho_ratio) / max(1e-6, inp.bc)
            ax2 = -k2 * mrx * vrel2
            ay2 = -G_FTPS2 - k2 * mry * vrel2
            az2 = -k2 * mrz * vrel2

        # Advance
        vx += ax2 * dt
        vy += ay2 * dt
        vz += az2 * dt
        x  += mx * dt
        y  += my * dt
        z  += mz * dt
        t  += dt

        while next_i < len(dists) and x >= dists[next_i] * FT_PER_YD:
            yd = dists[next_i]
            y_rel_sight_ft = y + scope_h_ft
            drop_in = -y_rel_sight_ft * IN_PER_FT
            drift_in = z * IN_PER_FT  # signed: +R, -L

            if inp.angular_units == "MIL":
                elev = to_mil(drop_in, yd)
                wind = to_mil(drift_in, yd)
            else:
                elev = to_moa(drop_in, yd)
                wind = to_moa(drift_in, yd)

            v_now = (vx*vx + vy*vy + vz*vz)**0.5
            rows.append(Row(
                yd=yd,
                elev=elev,
                wind=wind,
                tof_s=t,
                vel_fps=v_now,
                energy_ftlb=energy_ftlb(v_now, bullet_gr=bullet_gr),
            ))
            next_i += 1

    return rows
