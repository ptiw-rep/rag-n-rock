from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_ENTITY
from util import logger

def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        f"HTTPException [{exc.status_code}] at {request.url.path}: {exc.detail}",
        exc_info=True
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error(
        f"SQLAlchemyError at {request.url.path}: {str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "A database error occurred."}
    )

def generic_exception_handler(request: Request, exc: Exception):
    logger.critical(
        f"Unhandled exception at {request.url.path}: {str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."}
    )
