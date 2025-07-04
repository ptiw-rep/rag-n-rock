import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from datetime import timedelta

from config import get_env
from util import logger
from database.db_session import get_db
from rag.models.data_schema import RegisterRequest
from database.models import User
from util.auth_handler import get_password_hash, authenticate_user,create_access_token,ACCESS_TOKEN_EXPIRE_MINUTES

log = logging.getLogger(__name__)

router = APIRouter()
SECRET_KEY = get_env("SECRET_KEY", "my-secret-key")
ALGORITHM = get_env("ALGORITHM", "HS256")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)):
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
            logger.warning("JWT missing 'sub' claim")
            raise credentials_exception
        
        logger.info(f"Token validated for user: {username}")
        return {"username": username}
    
    except JWTError:
        logger.error(f"JWT decoding error.", exc_info=True)
        raise credentials_exception
    
@router.post("/register")
def register_user(request : RegisterRequest, db: Session = Depends(get_db)):
    """_summary_

    Args:
        request (RegisterRequest): Registratioon Request containing username and password. Ideally passwords should be hashed before transit.
        db (Session, optional): _description_. Defaults to Depends(get_db).

    Raises:
        HTTPException: If the username already exists in the database.

    Returns:
        dict: Response message indicating success or failure of user registration
        
    """
    logger.info(f"Registration request received for username: {request.username}")

    existing_user = db.query(User).filter(User.username == request.username).first()

    if existing_user:
        logger.warning(f"Username '{request.username}' already taken")
        raise HTTPException(status_code=400, detail="Username already taken")

    try:
        hashed_pw = get_password_hash(request.password)
        logger.debug("Password hashed successfully")

        new_user = User(username=request.username, password_hash=hashed_pw)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        logger.info(f"User registered successfully: {request.username}")
        return {"message": "User created successfully"}
    
    except Exception as e:
        logger.error(f"Error during user registration: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="User registration failed")

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """_summary_

    Args:
        form_data (OAuth2PasswordRequestForm, optional): _description_. Defaults to Depends().
        db (Session, optional): _description_. Defaults to Depends(get_db).

    Raises:
        HTTPException: If authentication fails due to incorrect username or password.

    Returns:
        dict: = Response containing access token and token type
    """
    logger.info(f"Login attempt for user: {form_data.username}")

    user = authenticate_user(db, form_data.username, form_data.password)

    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        access_token_expires = timedelta(minutes=int(ACCESS_TOKEN_EXPIRE_MINUTES))
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        logger.info(f"Login successful for user: {form_data.username}")
        return {"access_token": access_token, "token_type": "bearer"}
    
    except Exception as e:
        logger.error(f"Error generating access token for {form_data.username}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Login failed")

