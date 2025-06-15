import logging
from ..data_plane.models.data_schema import RegisterRequest
from sqlalchemy.orm import Session

log  = logging.getLogger(__name__)

def register_user(db: Session, data : RegisterRequest):
    log.info(f"Attemping to register user: {data.username} ")
    
def login_user(db: Session, data : RegisterRequest):
    log.info(f"Attempting to login user: {data.username} ")