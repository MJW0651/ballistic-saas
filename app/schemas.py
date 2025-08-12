from pydantic import BaseModel, field_validator

class SolveInput(BaseModel):
    zero_yd: int
    scope_h_in: float
    mv_fps: float
    bc: float
    bc_model: str  # "G7" or "G1"
    temp_f: float
    pressure_inHg: float
    rh: float
    altitude_ft: float
    wind_speed_mph: float
    wind_dir_deg: float
    start_yd: int
    end_yd: int
    step_yd: int
    angular_units: str   # "MIL" or "MOA"

    @field_validator("angular_units")
    @classmethod
    def valid_units(cls, v):
        if v not in ("MIL", "MOA"):
            raise ValueError("angular_units must be MIL or MOA")
        return v

    @field_validator("bc_model")
    @classmethod
    def valid_bc_model(cls, v):
        if v.upper() not in ("G1", "G7"):
            raise ValueError("bc_model must be G1 or G7")
        return v.upper()
