from dataclasses import dataclass
from math import radians, sin
import numpy as np

from .schemas import SolveInput

# Constants
G_FTPS2 = 32.174
IN_PER_FT = 12.0
FT_PER_YD = 3.0
FTPS_PER_MPH = 1.4666667

@dataclass
class Row:
    yd: int
    elev: float
    wind: float
    tof_s: float
    vel_fps: float
    energy_ftlb: float

def to_mil(inches: float, yards: float) -> float:
    return round((inches / (yards * 36.0)) * 1000.0, 1)

def to_moa(inches: float, yards: float) -> float:
    moa = inches / (1.0472 * (yards / 100.0))
    return round(moa * 4.0) / 4.0

def energy_ftlb(vel_fps: float, bullet_gr: float = 140.0) -> float:
    # E(ft-lb) = (w(gr) * v^2) / 450240
    return (bullet_gr * vel_fps**2) / 450240.0

def compute_table(inp: SolveInput):
    """
    Minimal no-drag model so the web flow works end-to-end.
    We'll replace with a G7/G1 solver later.
    """
    v0 = float(inp.mv_fps)
    zero_ft = inp.zero_yd * FT_PER_YD
    scope_h_ft = inp.scope_h_in / IN_PER_FT

    # Choose a tiny muzzle angle to achieve the requested zero (ignoring drag)
    theta = ((G_FTPS2 * zero_ft**2) / (2.0 * v0**2) - scope_h_ft) / zero_ft

    dists = np.arange(inp.start_yd, inp.end_yd + 1, inp.step_yd, dtype=int)
    rows = []

    # Wind full-value component (simplified)
    wind_full_mph = inp.wind_speed_mph * abs(sin(radians(inp.wind_dir_deg)))
    wind_ftps = wind_full_mph * FTPS_PER_MPH

    for yd in dists:
        x_ft = yd * FT_PER_YD
        tof = x_ft / v0
        y_bullet_ft = x_ft * theta - (G_FTPS2 * x_ft**2) / (2.0 * v0**2)
        y_rel_sight_ft = y_bullet_ft + scope_h_ft
        drop_in = -y_rel_sight_ft * IN_PER_FT
        drift_in = wind_ftps * tof * IN_PER_FT

        if inp.angular_units == "MIL":
            elev = to_mil(drop_in, yd)
            wind = to_mil(drift_in, yd)
        else:
            elev = to_moa(drop_in, yd)
            wind = to_moa(drift_in, yd)

        rows.append(Row(
            yd=int(yd),
            elev=elev,
            wind=wind,
            tof_s=tof,
            vel_fps=v0,
            energy_ftlb=energy_ftlb(v0)
        ))
    return rows
