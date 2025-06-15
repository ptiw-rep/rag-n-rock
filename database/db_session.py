from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
import os
from rag import CHROMA_PATH

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/chat_rag.db"))

os.makedirs(CHROMA_PATH, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        print("[init_db] Database initialized successfully.")
    except ImportError:
        print("[init_db] ImportError: Ensure all models are correctly defined and imported.")
        raise
    except Exception as e:    
        print(f"[init_db] Database initialization failed: {e}")
        raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
