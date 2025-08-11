import os
import datetime
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
def safe_json_dumps(v):
    """Safely serialize objects to JSON, handling Decimal types and other edge cases"""
    import json
    from decimal import Decimal
    
    def convert_decimals(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_decimals(item) for item in obj]
        return obj
    
    try:
        converted = convert_decimals(v)
        return json.dumps(converted, default=str)
    except Exception as e:
        return f"Error serializing: {str(e)}"

templates.env.filters["tojson"] = safe_json_dumps

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

logger = getLogger("app")

async def _initialize_data_source():
    """Initialize data source if startup didn't complete"""
    logger.info("Attempting to initialize data source...")
    
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
    
    # Check deployment environment
    is_render = os.getenv("RENDER") == "true"
    logger.info(f"Running on Render: {is_render}")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return
    
    app.state.dataset_source = None
    app.state.dataset_table = None
    
    # Connect to Neon database with retry logic for Render
    max_retries = 3 if is_render else 1
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Database connection attempt {attempt + 1}/{max_retries}")
            
            from sqlalchemy import text
            with SessionLocal() as db:
                # Test database connection
                try:
                    db.execute(text("SELECT 1")).scalar()
                    logger.info("Neon database connection successful")
                except Exception as e:
                    logger.error(f"Neon database connection test failed: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        logger.error("All database connection attempts failed")
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
                        break  # Success, exit retry loop
                    else:
                        logger.error(f"Table {table_name} exists but is empty")
                        break  # No point retrying if table is empty
                except Exception as e:
                    logger.error(f"Error accessing table {table_name}: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        logger.error("All table access attempts failed")
                        break
                        
        except Exception as e:
            logger.error(f"Database setup attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                continue
            else:
                logger.error("All database setup attempts failed")
                break
    
    if not app.state.dataset_source:
        logger.error("Failed to connect to Neon database - application will not work correctly")
        # Don't try CSV fallback on Render since it won't work
        if not is_render and pd is not None:
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
                logger.warning(f"Failed to load CSV data: {exc}")
                app.state.df = None
        else:
            logger.warning("CSV fallback not available on Render")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("base.html", {"request": request, "title": "Home"})

@app.get("/health")
async def health_check():
    """Health check endpoint to verify application state"""
    try:
        # Check database connection
        with SessionLocal() as db:
            from sqlalchemy import text
            db.execute(text("SELECT 1")).scalar()
            db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Check app state
    app_state = {
        "dataset_source": getattr(app.state, "dataset_source", None),
        "dataset_table": getattr(app.state, "dataset_table", None),
        "has_dataframe": getattr(app.state, "df", None) is not None,
        "database_status": db_status,
        "render_env": os.getenv("RENDER") == "true"
    }
    
    return {
        "status": "healthy" if app_state["dataset_source"] else "unhealthy",
        "app_state": app_state,
        "timestamp": str(datetime.datetime.now())
    }

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
        # Convert Decimal types to float for JSON serialization
        data = []
        for row in rows:
            row_dict = dict(row)
            # Convert Decimal values to float
            for key, value in row_dict.items():
                if hasattr(value, 'as_tuple'):  # Check if it's a Decimal
                    row_dict[key] = float(value)
            data.append(row_dict)
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
    # Get database preview data
    table_preview = None
    try:
        source = getattr(app.state, "dataset_source", None)
        table = getattr(app.state, "dataset_table", None)
        
        if source == "db" and table:
            with SessionLocal() as db:
                from sqlalchemy import text
                # Get first 10 rows for preview
                sql = f'SELECT * FROM "{table}" LIMIT 10'
                rows = db.execute(text(sql)).mappings().all()
                if rows:
                    # Convert Decimal types to float for JSON serialization
                    preview_data = []
                    for row in rows:
                        row_dict = dict(row)
                        for key, value in row_dict.items():
                            if hasattr(value, 'as_tuple'):  # Check if it's a Decimal
                                row_dict[key] = float(value)
                        preview_data.append(row_dict)
                    
                    # Get column names
                    cols = list(rows[0].keys()) if rows else []
                    table_preview = {
                        "data": preview_data,
                        "columns": cols,
                        "total_rows": None  # We'll get this separately
                    }
                    
                    # Get total row count
                    count_sql = f'SELECT COUNT(*) FROM "{table}"'
                    total_count = db.execute(text(count_sql)).scalar()
                    if hasattr(total_count, 'as_tuple'):  # Convert Decimal if needed
                        total_count = float(total_count)
                    table_preview["total_rows"] = total_count
                    
    except Exception as e:
        logger.error(f"Failed to get table preview: {e}")
        table_preview = None
    
    return templates.TemplateResponse("insights.html", {
        "request": request, 
        "title": "Insights", 
        "result": None,
        "table_preview": table_preview
    })

@app.post("/dashboard", response_class=HTMLResponse)
async def insights_post(request: Request, question: str = Form(...)):
    # Debug: Check app state
    logger.info(f"App state - dataset_source: {getattr(app.state, 'dataset_source', None)}")
    logger.info(f"App state - dataset_table: {getattr(app.state, 'dataset_table', None)}")
    logger.info(f"App state - df: {getattr(app.state, 'df', None)}")
    
    source = getattr(app.state, "dataset_source", None)
    table = getattr(app.state, "dataset_table", None)
    
    logger.info(f"Processing dashboard request - Data source: {source}, Table: {table}")
    
    # If startup didn't complete, try to initialize now
    if not source and not table:
        logger.warning("Startup incomplete, attempting to initialize database connection now")
        try:
            await _initialize_data_source()
            source = getattr(app.state, "dataset_source", None)
            table = getattr(app.state, "dataset_table", None)
            logger.info(f"Re-initialized - Data source: {source}, Table: {table}")
        except Exception as e:
            logger.error(f"Failed to re-initialize: {e}")
    
    if source == "db" and table:
        logger.info(f"Using database source: {table}")
        try:
            with SessionLocal() as db:
                result = basic_query_to_agg_db(question, db, table)
                logger.info(f"DB query result: {result}")
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            result = {"data": [], "groupby": None, "metric": None}
    else:
        logger.warning(f"Database not available - source: {source}, table: {table}")
        df = getattr(app.state, "df", None)
        if df is not None:
            logger.info(f"Using CSV data source. DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        else:
            logger.warning("No DataFrame available")
        result = basic_query_to_agg_csv(question, df)
        logger.info(f"CSV query result: {result}")
    
    # Only generate summary if we have data
    if result and result.get('data'):
        try:
            summary_text = summarize_text(f"Create a one-paragraph summary for this data grouped by {result['groupby']} with values {result['metric']}: {result['data']}")
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            summary_text = "Unable to generate summary due to an error."
    else:
        summary_text = "No data available to summarize. Please check the data source configuration."
    
    # Get table preview for display
    table_preview = None
    try:
        source = getattr(app.state, "dataset_source", None)
        table = getattr(app.state, "dataset_table", None)
        
        if source == "db" and table:
            with SessionLocal() as db:
                from sqlalchemy import text
                # Get first 10 rows for preview
                sql = f'SELECT * FROM "{table}" LIMIT 10'
                rows = db.execute(text(sql)).mappings().all()
                if rows:
                    # Convert Decimal types to float for JSON serialization
                    preview_data = []
                    for row in rows:
                        row_dict = dict(row)
                        for key, value in row_dict.items():
                            if hasattr(value, 'as_tuple'):  # Check if it's a Decimal
                                row_dict[key] = float(value)
                        preview_data.append(row_dict)
                    
                    # Get column names
                    cols = list(rows[0].keys()) if rows else []
                    table_preview = {
                        "data": preview_data,
                        "columns": cols,
                        "total_rows": None
                    }
                    
                    # Get total row count
                    count_sql = f'SELECT COUNT(*) FROM "{table}"'
                    total_count = db.execute(text(count_sql)).scalar()
                    if hasattr(total_count, 'as_tuple'):  # Convert Decimal if needed
                        total_count = float(total_count)
                    table_preview["total_rows"] = total_count
                    
    except Exception as e:
        logger.error(f"Failed to get table preview: {e}")
        table_preview = None
    
    return templates.TemplateResponse("insights.html", {
        "request": request, 
        "title": "Insights", 
        "result": result, 
        "narrative": summary_text, 
        "question": question,
        "table_preview": table_preview
    })