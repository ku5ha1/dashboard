import os
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
    logger.info("Starting application initialization...")
    
    # Check if running on Vercel
    is_vercel = os.getenv("VERCEL") == "1"
    logger.info(f"Running on Vercel: {is_vercel}")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return
    
    app.state.dataset_source = None
    app.state.dataset_table = None
    
    # Connect to Neon database
    try:
        from sqlalchemy import text
        with SessionLocal() as db:
            # Test database connection
            try:
                db.execute(text("SELECT 1")).scalar()
                logger.info("Neon database connection successful")
            except Exception as e:
                logger.error(f"Neon database connection failed: {e}")
                return
            
            # We know we're using the education table
            table_name = "education"
            try:
                # Check if table has data
                count = db.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
                logger.info(f"Found {count} rows in {table_name} table")
                
                if count > 0:
                    # Get column names
                    cols = db.execute(text(f'SELECT * FROM "{table_name}" LIMIT 0')).keys()
                    logger.info(f"Table columns: {cols}")
                    
                    # Set as data source
                    app.state.dataset_source = "db"
                    app.state.dataset_table = table_name
                    logger.info(f"Successfully connected to Neon database table '{table_name}'")
                else:
                    logger.error(f"Table {table_name} exists but is empty")
            except Exception as e:
                logger.error(f"Error accessing table {table_name}: {e}")
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
    
    if not app.state.dataset_source:
        logger.error("Failed to connect to Neon database - application will not work correctly")
    if pd is not None:
        csv_path = _resolve_csv_path(settings.DATA_CSV_PATH)
        try:
            app.state.df = pd.read_csv(csv_path)
            app.state.dataset_source = "csv"
            logger.info(f"Loaded CSV dataset: {csv_path} shape={getattr(app.state.df, 'shape', None)}")
            if app.state.df is not None and not app.state.df.empty:
                logger.info(f"CSV data loaded successfully. Columns: {list(app.state.df.columns)}")
            else:
                logger.warning("CSV data loaded but DataFrame is empty")
        except Exception as exc:
            logger.warning(f"Failed to load dataset from {csv_path}: {exc}")
            app.state.df = None
    else:
        logger.warning("Pandas is not available - CSV data source will not work")

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
    # Update candidates to match actual column names
    candidate_gbs = ["state", "gender", "class", "school_name"]
    gb = None
    
    # Map common terms to actual column names
    column_mapping = {
        "school": "school_name",
        "score": "marks",
        "grade": "marks",
        "performance": "marks"
    }
    
    # First try exact matches
    for c in candidate_gbs:
        if c in q_low:
            gb = c
            break
    
    # Then try mapped terms
    if gb is None:
        for term, col in column_mapping.items():
            if term in q_low and col in candidate_gbs:
                gb = col
                break
    
    if gb is None:
        gb = available_cols[0] if available_cols else "state"
        
    # Always use marks as the metric if available
    metric = "marks" if (available_cols and "marks" in available_cols) or any(term in q_low for term in ["marks", "score", "grade", "performance"]) else None
    
    logger.info(f"Chose grouping '{gb}' and metric '{metric}' from query: {q}")
    return {"gb": gb, "metric": metric}

def basic_query_to_agg_db(q: str, session: Session, table_name: str) -> Dict[str, Any]:
    from sqlalchemy import text
    
    # Define known columns and their metrics
    COLUMNS = {
        "state": "State distribution",
        "gender": "Gender distribution",
        "class": "Class distribution",
        "school_name": "School distribution",
        "marks": "Student marks",
        "course_name": "Course distribution",
        "completion_status": "Completion status"
    }
    
    # Get query intent
    q_low = q.lower()
    gb = None
    metric = None
    
    # Try to find grouping column from query
    for col, description in COLUMNS.items():
        if any(term in q_low for term in [col.lower(), description.lower()]):
            gb = col
            break
    
    # Default to state if no grouping found
    if not gb:
        gb = "state"
        logger.info(f"No specific grouping found in query, defaulting to {gb}")
    
    # Determine if we should show marks or counts
    if any(term in q_low for term in ["marks", "score", "performance", "grade", "average"]):
        metric = "marks"
    
    logger.info(f"Analyzing by {gb}" + (f" with {metric} values" if metric else " with counts"))
    
    try:
        # Build and execute query
        if metric:
            sql = f'''
                SELECT 
                    "{gb}" as "{gb}",
                    ROUND(AVG(CAST("{metric}" as FLOAT))::NUMERIC, 2) as value,
                    COUNT(*) as count
                FROM "{table_name}"
                WHERE "{metric}" IS NOT NULL
                GROUP BY "{gb}"
                ORDER BY value DESC
            '''
        else:
            sql = f'''
                SELECT 
                    "{gb}" as "{gb}",
                    COUNT(*) as value
                FROM "{table_name}"
                GROUP BY "{gb}"
                ORDER BY value DESC
            '''
        
        logger.info(f"Executing query: {sql}")
        rows = session.execute(text(sql)).mappings().all()
        data = [dict(r) for r in rows]
        logger.info(f"Query returned {len(data)} rows")
        
        if not data:
            logger.warning("Query returned no data")
            return {"data": [], "groupby": None, "metric": None}
            
        return {
            "data": data, 
            "groupby": gb, 
            "metric": metric or "count",
            "description": COLUMNS.get(gb, gb)
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {"data": [], "groupby": None, "metric": None}

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
    logger.info(f"Data source: {source}, Table: {getattr(app.state, 'dataset_table', None)}")
    if source == "db" and app.state.dataset_table:
        with SessionLocal() as db:
            result = basic_query_to_agg_db(question, db, app.state.dataset_table)
            logger.info(f"DB query result: {result}")
    else:
        df = getattr(app.state, "df", None)
        if df is not None:
            logger.info(f"Using CSV data source. DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        else:
            logger.warning("No DataFrame available")
        result = basic_query_to_agg_csv(question, df)
        logger.info(f"CSV query result: {result}")
    # Optional: ask LLM to narrate
    summary_text = summarize_text(f"Create a one-paragraph summary for this data grouped by {result['groupby']} with values {result['metric']}: {result['data']}")
    return templates.TemplateResponse("insights.html", {"request": request, "title": "Insights", "result": result, "narrative": summary_text, "question": question})