from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from passlib.context import CryptContext
from jose import JWTError, jwt

from database.models import User as DBUser
from database.db_session import get_db

from config import get_env
from util import logger

# Configuration
SECRET_KEY = get_env("SECRET_KEY", "my-secret-key")
ALGORITHM = get_env("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = get_env("ACCESS_TOKEN_EXPIRE_MINUTES", 30)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"])

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def verify_password(plain_password, hashed_password):
    logger.debug("Verifying password hash")
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    logger.debug("Hashing password")
    return pwd_context.hash(password)

def get_user(db: Session, username: str):
    logger.info(f"Fetching user from database: {username}")
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if not user:
        logger.warning(f"User not found: {username}")
    return user

def authenticate_user(db: Session, username: str, password: str):
    logger.info(f"Authenticating user: {username}")
    user = get_user(db, username)
    if not user:
        logger.warning(f"Authentication failed for {username}: User not found")
        return False
    if not verify_password(password, user.password_hash):
        logger.warning(f"Authentication failed for {username}: Incorrect password")
        return False
    logger.info(f"User authenticated successfully: {username}")
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    logger.debug("Creating access token")
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.info("Access token created successfully")
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    logger.debug("Validating access token")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Token missing 'sub' claim")
            raise credentials_exception
    except JWTError:
        logger.error(f"JWT decoding error: {str(e)}", exc_info=True)
        raise credentials_exception
    
    user = get_user(db, username=username)
    if user is None:
        logger.warning(f"User not found in DB: {username}")
        raise credentials_exception
    return user