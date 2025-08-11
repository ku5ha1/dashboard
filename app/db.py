from sqlalchemy import create_engine, Column, Integer, Text, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base
from pathlib import Path
from logging import getLogger
from typing import Optional
from .config import settings

logger = getLogger("app.db")

def _normalize_database_url(database_url: str) -> str:
    """Normalize database URL for Neon and ensure proper driver usage."""
    logger.info("Normalizing database URL...")
    
    if not database_url:
        logger.error("No DATABASE_URL provided")
        raise ValueError("DATABASE_URL environment variable is required")
        
    # Convert postgres:// to postgresql://
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]
        logger.info("Converted postgres:// to postgresql://")
    
    # Add psycopg driver if not present
    if database_url.startswith("postgresql://") and "+psycopg" not in database_url and "+psycopg2" not in database_url:
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        logger.info("Added psycopg driver to URL")
    
    # Ensure SSL mode for Neon
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
        logger.info("Added SSL mode requirement")
    
    # Log sanitized URL
    url_parts = database_url.split('@')
    if len(url_parts) > 1:
        masked_url = f"{url_parts[0].split('://')[0]}://[CREDENTIALS]@{url_parts[1]}"
    else:
        masked_url = "[MALFORMED_URL]"
    logger.info(f"Using database URL: {masked_url}")
    
    return database_url

def _resolve_sqlite_url(database_url: str) -> str:
    if not database_url.startswith("sqlite"):
        return database_url

    # Extract filesystem path
    fs_path = None
    if database_url.startswith("sqlite:////"):  # absolute
        fs_path = Path(database_url[len("sqlite:////"):])
    elif database_url.startswith("sqlite:///"):  # relative
        fs_path = Path(database_url[len("sqlite///"):])
        if str(fs_path).startswith(" "):
            fs_path = Path(str(fs_path).strip())
        if not fs_path.is_absolute():
            fs_path = (Path(__file__).parent.parent / fs_path).resolve()
    else:
        return database_url

    # Try to ensure directory exists and is writable; fall back to /tmp on failure (Vercel)
    try:
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        # Test write permission
        with open(fs_path, "a", encoding="utf-8") as _:
            pass
        resolved = f"sqlite:////{fs_path.as_posix()}"
        logger.info(f"Using SQLite at {resolved}")
        return resolved
    except Exception as exc:
        tmp_path = Path("/tmp/app.db")
        try:
            with open(tmp_path, "a", encoding="utf-8") as _:
                pass
            resolved = f"sqlite:////{tmp_path.as_posix()}"
            logger.warning(f"Falling back to {resolved} due to: {exc}")
            return resolved
        except Exception as exc2:
            logger.error(f"Failed to prepare SQLite file: {exc2}")
            return database_url


# Normalize and validate database URL
try:
    normalized_url = _normalize_database_url(settings.DATABASE_URL)
    logger.info("Database URL normalized successfully")
except Exception as e:
    logger.error(f"Failed to normalize database URL: {e}")
    raise

# Create engine with proper configuration for Neon
try:
    engine = create_engine(
        normalized_url,
        pool_size=5,  # Limit connections for serverless
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,  # Recycle connections every 30 minutes
        pool_pre_ping=True,  # Verify connection before using
        echo=False  # Set to True for SQL query logging
    )
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Summary(Base):
    __tablename__ = "summaries"
    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(Text, nullable=False)
    summary_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

def init_db():
    Base.metadata.create_all(bind=engine)

# Optional: simple Blob-based persistence helpers (for Vercel)
try:
    from vercel_blob import put, list as blob_list, get as blob_get
    import json
    import os

    def save_summary_blob(input_text: str, summary_text: str) -> Optional[str]:
        token = settings.BLOB_READ_WRITE_TOKEN
        if not token:
            return None
        payload = {
            "input_text": input_text,
            "summary_text": summary_text,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # add random suffix server-side
        resp = put("summaries/summary.json", data, {"access": "private", "addRandomSuffix": True, "contentType": "application/json"})
        return resp.get("url")
except Exception:
    # Blob SDK not available; skip
    def save_summary_blob(input_text: str, summary_text: str) -> Optional[str]:
        return None