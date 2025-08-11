from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import pandas as pd
from typing import Optional, Dict, Any

from .db import SessionLocal, init_db, Summary
from .services.llm import summarize_text, make_revision_notes
from .config import settings

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup():
    init_db()
    try:
        app.state.df = pd.read_csv(settings.DATA_CSV_PATH)
    except Exception:
        app.state.df = pd.DataFrame()

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
    row = Summary(input_text=input_text, summary_text=summary)
    db.add(row); db.commit(); db.refresh(row)
    return templates.TemplateResponse("summariser.html", {"request": request, "title": "Summariser", "summary": summary, "saved_id": row.id, "input_text": input_text})

# Revise
@app.get("/revise", response_class=HTMLResponse)
async def revise_get(request: Request, db: Session = Depends(get_db)):
    items = db.query(Summary).order_by(Summary.id.desc()).limit(20).all()
    return templates.TemplateResponse("revise.html", {"request": request, "title": "Revise", "items": items})

@app.post("/revise/{summary_id}", response_class=HTMLResponse)
async def revise_generate(request: Request, summary_id: int, db: Session = Depends(get_db)):
    row = db.query(Summary).filter(Summary.id == summary_id).first()
    if not row:
        return RedirectResponse(url="/revise", status_code=302)
    notes = make_revision_notes(row.summary_text)
    return templates.TemplateResponse("revise.html", {"request": request, "title": "Revise", "items": db.query(Summary).order_by(Summary.id.desc()).limit(20).all(), "notes": notes, "active_id": summary_id})

# Insights (minimal)
def basic_query_to_agg(df: pd.DataFrame, q: str) -> Dict[str, Any]:
    if df.empty:
        return {"data": [], "groupby": None, "metric": None}
    q_low = q.lower()
    metric = "score" if "score" in df.columns else None
    if "state" in q_low:
        gb = "state"
    elif "gender" in q_low:
        gb = "gender"
    elif "class" in q_low:
        gb = "class"
    else:
        gb = "state" if "state" in df.columns else df.columns[0]
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
    df = getattr(app.state, "df", None)
    result = basic_query_to_agg(df, question) if df is not None else {"data": [], "groupby": None, "metric": None}
    # Optional: ask LLM to narrate
    summary_text = summarize_text(f"Create a one-paragraph summary for this data grouped by {result['groupby']} with values {result['metric']}: {result['data']}")
    return templates.TemplateResponse("insights.html", {"request": request, "title": "Insights", "result": result, "narrative": summary_text, "question": question})