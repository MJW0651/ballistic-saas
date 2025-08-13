from dataclasses import dataclass
from math import radians, sin, cos
import json
import pathlib

from .schemas import SolveInput

# ----------------- Constants -----------------
G_FTPS2 = 32.174             # gravity (ft/s^2)
IN_PER_FT = 12.0
FT_PER_YD = 3.0
FTPS_PER_MPH = 1.4666667     # mph -> ft/s

# Cache for drag tables (loaded once)
_DRAG = {}

@dataclass
class Row:
    yd: int
    elev: float          # MIL or MOA (per input selection)
    wind: float          # MIL or MOA (signed: +R, -L)
    tof_s: float         # time of flight (s)
    vel_fps: float       # remaining (ground-relative) velocity (ft/s)
    energy_ftlb: float   # kinetic energy (ft-lb)

# ----------------- Unit + energy helpers -----------------
def to_mil(inches: float, yards: float) -> float:
    """MIL hold given linear offset in inches at range in yards (2 decimals)."""
    return round((inches / (yards * 36.0)) * 1000.0, 2)

def to_moa(inches: float, yards: float) -> float:
    """MOA hold given linear offset in inches at range in yards (0.1 MOA)."""
    moa = inches / (1.0472 * (yards / 100.0))
    return round(moa, 1)

def energy_ftlb(vel_fps: float, bullet_gr: float = 140.0) -> float:
    # E (ft-lb) = (weight_gr * v^2) / 450240
    return (bullet_gr * vel_fps**2) / 450240.0

# ----------------- Drag model helpers -----------------
def _load_drag(model: str):
    """
    Load and cache a drag table for G1 or G7.
    File is JSON array of [fps, drag_coeff] in descending velocity.
    """
    key = model.upper()
    if key not in ("G1", "G7"):
        key = "G7"
    if key not in _DRAG:
        here = pathlib.Path(__file__).parent
        p = here / "drag" / f"{key.lower()}.json"
        _DRAG[key] = json.loads(p.read_text())
    return _DRAG[key]

def _interp_drag(v_fps: float, tbl):
    """
    Piecewise-linear interpolation on the drag table.
    If outside the table bounds, clamp to nearest endpoint.
    """
    if v_fps >= tbl[0][0]:
        return tbl[0][1]
    if v_fps <= tbl[-1][0]:
        return tbl[-1][1]
    for (v1, d1), (v2, d2) in zip(tbl, tbl[1:]):
        if v2 <= v_fps <= v1:
            t = (v1 - v_fps) / (v1 - v2)
            return d1 + t * (d2 - d1)
    return tbl[-1][1]

def _air_density_scale(temp_f: float, pressure_inHg: float, rh: float, altitude_ft: float) -> float:
    """
    First-order density scaling:
      rho ~ (pressure / 29.92) * (519.67 / T_R)
    Simple and effective for practical weather changes.
    """
    T_R = temp_f + 459.67  # Rankine
    return (max(0.1, pressure_inHg) / 29.92) * (519.67 / max(200.0, T_R))

# ----------------- Main solver -----------------
def compute_table(inp: SolveInput):
    """
    RK2 (midpoint) integrator with selectable G1/G7 drag.

    Coordinates:
      x: forward (ft), y: up (ft), z: right (ft)
      vx, vy, vz are ground-relative (ft/s)

    Drag is computed from air-relative velocity:
      v_rel = v_bullet - v_wind
      a_drag = -C(v_rel) * v_rel    (per component)
    Gravity acts on y.

    Wind convention (wind_dir_deg is where wind COMES FROM):
      0° = headwind (from target -> shooter), 180° = tailwind,
      90° = from right (blows right->left), 270° = from left (blows left->right).

    Wind hold sign:
      z > 0 means bullet drifted RIGHT. We output +R, -L (numeric sign only here).
    """
    # Load drag model
    tbl = _load_drag(inp.bc_model)

    # Inputs & derived values
    v0 = float(inp.mv_fps)
    zero_ft = inp.zero_yd * FT_PER_YD
    scope_h_ft = inp.scope_h_in / IN_PER_FT
    dens = _air_density_scale(inp.temp_f, inp.pressure_inHg, inp.rh, inp.altitude_ft)

    # Muzzle angle seed from a no-drag zero (integration then dominates)
    theta = ((G_FTPS2 * zero_ft**2) / (2.0 * v0**2) - scope_h_ft) / max(1.0, zero_ft)

    # Initial bullet state (ground-relative)
    vx, vy, vz = v0, v0 * theta, 0.0
    x = y = z = 0.0
    t = 0.0

    # Distances to record (yards -> feet)
    dists = list(range(inp.start_yd, inp.end_yd + 1, inp.step_yd))
    next_i = 0

    # --- Wind vector (ft/s), based on "coming from" angle ---
    # 0°: headwind (from +x toward -x) => wind vector ~ -x
    # 90°: from right => wind vector ~ -z
    wind_speed = max(0.0, float(inp.wind_speed_mph)) * FTPS_PER_MPH
    ang = radians(float(inp.wind_dir_deg))
    wind_x = -wind_speed * cos(ang)   # headwind reduces bullet's x; tailwind increases
    wind_z = -wind_speed * sin(ang)   # from right (90°) => -z; from left (270°) => +z

    # Time step based on solver quality
    dt = 0.001 if getattr(inp, "solver_quality", "standard") == "high" else 0.002
    bullet_gr = float(getattr(inp, "bullet_gr", 140.0))

    rows = []

    # Integrate until we've passed the last distance or timing out
    while next_i < len(dists) and t < 6.0 and vx > 0.1:
        # Air-relative components at current state
        rx = vx - wind_x
        ry = vy                     # no vertical wind in this model
        rz = vz - wind_z
        vrel = max(1.0, (rx*rx + ry*ry + rz*rz) ** 0.5)

        # Drag factor from table at |v_rel|
        drag = _interp_drag(vrel, tbl) * dens / max(1e-6, inp.bc)

        # Accelerations (drag per air-relative component; gravity on y)
        ax = -drag * rx
        ay = -G_FTPS2 - drag * ry
        az = -drag * rz

        # Midpoint (RK2): estimate state at t + dt/2
        mx = vx + ax * dt * 0.5
        my = vy + ay * dt * 0.5
        mz = vz + az * dt * 0.5

        # Air-relative at midpoint
        mrx = mx - wind_x
        mry = my
        mrz = mz - wind_z
        vrel2 = max(1.0, (mrx*mrx + mry*mry + mrz*mrz) ** 0.5)
        drag2 = _interp_drag(vrel2, tbl) * dens / max(1e-6, inp.bc)

        ax2 = -drag2 * mrx
        ay2 = -G_FTPS2 - drag2 * mry
        az2 = -drag2 * mrz

        # Advance bullet state (ground-relative)
        vx += ax2 * dt
        vy += ay2 * dt
        vz += az2 * dt
        x  += mx * dt
        y  += my * dt
        z  += mz * dt
        t  += dt

        # Emit rows when passing each requested distance
        while next_i < len(dists) and x >= dists[next_i] * FT_PER_YD:
            yd = dists[next_i]
            y_rel_sight_ft = y + scope_h_ft
            drop_in = -y_rel_sight_ft * IN_PER_FT
            drift_in = z * IN_PER_FT  # signed: +right, -left

            if inp.angular_units == "MIL":
                elev = to_mil(drop_in, yd)
                wind = to_mil(drift_in, yd)
            else:
                elev = to_moa(drop_in, yd)
                wind = to_moa(drift_in, yd)

            v_now = max(1.0, (vx*vx + vy*vy + vz*vz) ** 0.5)  # ground-relative speed
            rows.append(Row(
                yd=yd,
                elev=elev,
                wind=wind,                 # signed numeric (R+ / L-)
                tof_s=t,
                vel_fps=v_now,
                energy_ftlb=energy_ftlb(v_now, bullet_gr=bullet_gr),
            ))
            next_i += 1

    return rows
