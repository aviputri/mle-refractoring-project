from dotenv import load_dotenv
from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
import os

load_dotenv()

DATABASE_URL = os.getenv("DB_CONN")
if DATABASE_URL is None:
    raise ValueError("DB Conn env variable is not set")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
class Base(DeclarativeBase):
    """Base class shared by the app's SQLAlchemy models."""
def get_db() -> Generator[Session]:
    with SessionLocal() as session:
        yield session