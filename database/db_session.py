import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
from util import logger
from rag import CHROMA_PATH

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/chat_rag.db"))

os.makedirs(CHROMA_PATH, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    logger.info("Initializing database schema...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database initialized successfully at: {DB_PATH}")
    except ImportError:
        logger.error("Failed to import models. Ensure all models are correctly defined and imported.", exc_info=True)
        raise
    except Exception as e:    
        logger.error(f"Database initialization failed: {str(e)}", exc_info=True)
        raise

def get_db():
    db = SessionLocal()
    logger.debug("New database session started.")
    try:
        yield db
    finally:
        db.close()
        logger.debug("Database session closed.")
