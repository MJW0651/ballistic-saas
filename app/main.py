from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .ballistics import compute_table
from .schemas import SolveInput

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/solve", response_class=HTMLResponse)
async def solve(
    request: Request,
    zero_yd: int = Form(...),
    scope_h_in: float = Form(...),
    mv_fps: float = Form(...),
    bc: float = Form(...),
    bc_model: str = Form(...),

    bullet_gr: float = Form(140.0),
    solver_quality: str = Form("standard"),

    temp_f: float = Form(...),
    pressure_inHg: float = Form(...),
    rh: float = Form(...),
    altitude_ft: float = Form(...),
    wind_speed_mph: float = Form(...),
    wind_dir_deg: float = Form(...),
    start_yd: int = Form(...),
    end_yd: int = Form(...),
    step_yd: int = Form(...),
    angular_units: str = Form(...),
):
    payload = SolveInput(
        zero_yd=zero_yd,
        scope_h_in=scope_h_in,
        mv_fps=mv_fps,
        bc=bc,
        bc_model=bc_model,
        bullet_gr=bullet_gr,
        solver_quality=solver_quality,
        temp_f=temp_f,
        pressure_inHg=pressure_inHg,
        rh=rh,
        altitude_ft=altitude_ft,
        wind_speed_mph=wind_speed_mph,
        wind_dir_deg=wind_dir_deg,
        start_yd=start_yd,
        end_yd=end_yd,
        step_yd=step_yd,
        angular_units=angular_units,
    )
    rows = compute_table(payload)
    return templates.TemplateResponse(
        "results_table.html",
        {"request": request, "rows": rows, "units": angular_units},
    )
