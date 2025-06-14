# backend/db.py
from sqlalchemy.orm import Session
from .models import SessionLocal # Assuming SessionLocal is defined in models.py and accessible

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
