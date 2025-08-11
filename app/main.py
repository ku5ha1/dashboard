from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
try:
    import pandas as pd  # optional; not installed in production
except Exception:
    pd = None
from typing import Optional, Dict, Any
from pathlib import Path
from logging import getLogger

from .db import SessionLocal, init_db, Summary, save_summary_blob
from .services.llm import summarize_text, make_revision_notes
from .config import settings

app = FastAPI()

# Resolve static directory relative to this file; mount only if exists (Vercel deploy safety)
_static_dir = (Path(__file__).parent / "static").resolve()
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
templates = Jinja2Templates(directory="app/templates")
templates.env.filters["tojson"] = lambda v: __import__("json").dumps(v)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

logger = getLogger("app")

def _resolve_csv_path(config_path: str) -> Path:
    path = Path(config_path)
    candidates = [path]
    if not path.is_absolute():
        # relative to app root
        candidates.append(Path(__file__).parent / path)
        # common layout: app/data/<file>
        candidates.append(Path(__file__).parent / "data" / path.name)
        # relative to repo root
        candidates.append(Path(__file__).parent.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path

@app.on_event("startup")
def startup():
    init_db()
    app.state.dataset_source = None
    app.state.dataset_table = None
    # Prefer DB; fall back to CSV if pandas is available
    try:
        from sqlalchemy import text
        with SessionLocal() as db:
            for table_name in ["education", "dataset", "data_education"]:
                try:
                    db.execute(text(f'SELECT 1 FROM {table_name} LIMIT 1'))
                    app.state.dataset_source = "db"
                    app.state.dataset_table = table_name
                    logger.info(f"Using DB table '{table_name}' for insights")
                    return
                except Exception:
                    continue
    except Exception:
        pass
    if pd is not None:
        csv_path = _resolve_csv_path(settings.DATA_CSV_PATH)
        try:
            app.state.df = pd.read_csv(csv_path)
            app.state.dataset_source = "csv"
            logger.info(f"Loaded CSV dataset: {csv_path} shape={getattr(app.state.df, 'shape', None)}")
        except Exception as exc:
            logger.warning(f"Failed to load dataset from {csv_path}: {exc}")
            app.state.df = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("base.html", {"request": request, "title": "Home"})

# Summariser
@app.get("/summariser", response_class=HTMLResponse)
async def summariser_get(request: Request):
    return templates.TemplateResponse("summariser.html", {"request": request, "title": "Summariser"})

@app.post("/summariser", response_class=HTMLResponse)
async def summariser_post(request: Request, input_text: str = Form(...), db: Session = Depends(get_db)):
    summary = summarize_text(input_text)
    saved_id = None
    save_error = None
    try:
        row = Summary(input_text=input_text, summary_text=summary)
        db.add(row)
        db.commit()
        db.refresh(row)
        saved_id = row.id
    except Exception as exc:
        db.rollback()
        # Fallback: save to Vercel Blob if configured
        blob_url = save_summary_blob(input_text, summary)
        if blob_url:
            save_error = f"Saved to Blob: {blob_url} (DB unavailable)"
        else:
            save_error = f"Could not save to DB (and Blob not configured). {exc}"
    return templates.TemplateResponse(
        "summariser.html",
        {"request": request, "title": "Summariser", "summary": summary, "saved_id": saved_id, "save_error": save_error, "input_text": input_text},
    )

# Revise
@app.get("/revise", response_class=HTMLResponse)
async def revise_get(request: Request, db: Session = Depends(get_db)):
    try:
        items = db.query(Summary).order_by(Summary.id.desc()).limit(20).all()
        error = None
    except Exception as exc:
        items, error = [], f"DB unavailable: {exc}"
    return templates.TemplateResponse("revise.html", {"request": request, "title": "Revise", "items": items, "error": error})

@app.post("/revise/{summary_id}", response_class=HTMLResponse)
async def revise_generate(request: Request, summary_id: int, db: Session = Depends(get_db)):
    try:
        row = db.query(Summary).filter(Summary.id == summary_id).first()
        if not row:
            return RedirectResponse(url="/revise", status_code=302)
        notes = make_revision_notes(row.summary_text)
        items = db.query(Summary).order_by(Summary.id.desc()).limit(20).all()
        error = None
    except Exception as exc:
        notes, items, error = None, [], f"DB unavailable: {exc}"
    return templates.TemplateResponse("revise.html", {"request": request, "title": "Revise", "items": items, "notes": notes, "active_id": summary_id, "error": error})

# Insights (minimal)
def _choose_groupby_and_metric(q: str, available_cols: Optional[list] = None) -> Dict[str, str]:
    q_low = (q or "").lower()
    candidate_gbs = ["state", "gender", "class", "school"]
    gb = None
    for c in candidate_gbs:
        if c in q_low:
            gb = c
            break
    if gb is None:
        gb = available_cols[0] if available_cols else "state"
    metric = "score" if (available_cols and "score" in available_cols) or ("score" in q_low) else None
    return {"gb": gb, "metric": metric}

def basic_query_to_agg_db(q: str, session: Session, table_name: str) -> Dict[str, Any]:
    from sqlalchemy import text
    # Restrict identifiers to safe set
    allowed_cols = {"state", "gender", "class", "school", "score", "students"}
    choice = _choose_groupby_and_metric(q, list(allowed_cols))
    gb = choice["gb"]
    metric = choice["metric"]
    if gb not in allowed_cols:
        return {"data": [], "groupby": None, "metric": None}
    if metric == "score":
        sql = f'SELECT "{gb}" as "{gb}", AVG("score") as value FROM {table_name} GROUP BY "{gb}" ORDER BY value DESC'
    else:
        sql = f'SELECT "{gb}" as "{gb}", COUNT(*) as value FROM {table_name} GROUP BY "{gb}" ORDER BY value DESC'
    rows = session.execute(text(sql)).mappings().all()
    data = [dict(r) for r in rows]
    return {"data": data, "groupby": gb, "metric": metric or "count"}

def basic_query_to_agg_csv(q: str, df) -> Dict[str, Any]:
    if pd is None or df is None or getattr(df, "empty", True):
        return {"data": [], "groupby": None, "metric": None}
    available_cols = list(df.columns)
    choice = _choose_groupby_and_metric(q, available_cols)
    gb = choice["gb"]
    metric = choice["metric"]
    if metric and metric in df.columns:
        agg = df.groupby(gb)[metric].mean().reset_index().rename(columns={metric: "value"})
    else:
        agg = df.groupby(gb).size().reset_index(name="value")
    return {"data": agg.to_dict(orient="records"), "groupby": gb, "metric": metric or "count"}

@app.get("/dashboard", response_class=HTMLResponse)
async def insights_get(request: Request):
    return templates.TemplateResponse("insights.html", {"request": request, "title": "Insights", "result": None})

@app.post("/dashboard", response_class=HTMLResponse)
async def insights_post(request: Request, question: str = Form(...)):
    source = getattr(app.state, "dataset_source", None)
    if source == "db" and app.state.dataset_table:
        with SessionLocal() as db:
            result = basic_query_to_agg_db(question, db, app.state.dataset_table)
    else:
        df = getattr(app.state, "df", None)
        result = basic_query_to_agg_csv(question, df)
    # Optional: ask LLM to narrate
    summary_text = summarize_text(f"Create a one-paragraph summary for this data grouped by {result['groupby']} with values {result['metric']}: {result['data']}")
    return templates.TemplateResponse("insights.html", {"request": request, "title": "Insights", "result": result, "narrative": summary_text, "question": question})