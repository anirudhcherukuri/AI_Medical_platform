from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.config import DATABASE_URL
from backend.database.models import Base

# Create SQLite Engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Initializes database tables.
    """
    Base.metadata.create_all(bind=engine)
    print(f"[Database] SQLite database tables initialized at {DATABASE_URL}")

def get_db():
    """
    FastAPI Dependency for DB Session management.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
