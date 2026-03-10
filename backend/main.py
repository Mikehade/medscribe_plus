import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError
from utils.logger import get_logger

from src.config.dependency_injection.container import Container

logger = get_logger()

from src.api.base.router import router as base_router
from src.api.scribe.router import router as scribe_router, ws_router as scribe_ws_router
# from src.infrastructure.middleware.dependencies import get_current_user

#  Load .env and pick the right Settings class
load_dotenv(find_dotenv())
FASTAPI_ENV = os.getenv("APP_ENV", "development").lower()
logger.info(FASTAPI_ENV)

from src.config.development import DevSettings
from src.config.staging import StagingSettings
from src.config.production import ProductionSettings


settings = {
    "development": DevSettings(),
    "staging":     StagingSettings(),
    "production":  ProductionSettings(),
}[FASTAPI_ENV]

logger.info(f"Using settings: {type(settings).__name__}")

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown.
    Initializes and cleans up resources.
    """
    # Startup
    logger.info("Starting up application...")
    # logger.info(f"Database URI: {settings.SQLALCHEMY_DATABASE_URI[:20]}...")

    # Initialize container
    container.init_resources()

    await container.redis_client().connect()

    yield

    # Shutdown
    logger.info("Shutting down application...")
    
    # shutdown
    await container.redis_client().disconnect()

    container.shutdown_resources()
    logger.info("Application shutdown complete")

import src.api.base.router
import src.api.scribe.router
container = Container()

container.wire(
    modules=[
        # base_router
        # "src.api.base.router.router"
        src.api.base.router,
        src.api.scribe.router,

    ]
)



api_app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
    # root_path="/ai",
    root_path=os.getenv("API_ROOT_PATH", ""),
    docs_url="/api/v1/docs" if FASTAPI_ENV != "production" else None,  # Disable docs in production
    # redoc_url="/redoc" if FASTAPI_ENV != "production" else None,
)

# app = api_app


# Attach container to app state for access in dependencies
api_app.container = container


# CORS
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    # allow_credentials=False,
    allow_credentials=True,  # Changed to True for better auth support
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Define the security scheme for Swagger UI
security_scheme = HTTPBearer(
    scheme_name="JWT Authentication",
    description="Enter your JWT token in the format: `your-token-here` (without 'Bearer' prefix)",
    bearerFormat="JWT"
)

# Public routes (no authentication required)

# Protected routes (authentication required - handled at router level)
api_app.include_router(
    base_router,
    prefix=f"{settings.API_V_STR}",
    tags=["Core"],
    # dependencies=[Depends(get_current_user)],
)

api_app.include_router(
    scribe_router,
    prefix=f"{settings.API_V_STR}",
    tags=["Scribe"],
    # dependencies=[Depends(get_current_user)],
)


api_app.include_router(
    scribe_ws_router,
    # prefix=f"{settings.API_V_STR}",
    # tags=["Scribe"],
)


# Health-check
@api_app.get("/", status_code=200, tags=["Health"])
def welcome() -> dict:
    """
    Health check endpoint.
    Returns basic application information.
    """
    return {
        "message": "Welcome to MedScribe Agent API",
        "version": settings.VERSION,
        "environment": FASTAPI_ENV,
        "status": "healthy"
    }

@api_app.get("/health", status_code=200, tags=["Health"])
async def health_check() -> dict:
    """
    Detailed health check endpoint.
    Can be extended to check database connectivity, etc.
    """
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": FASTAPI_ENV,
    }

# Exception Handlers
@api_app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """
    Handle FastAPI request validation errors.
    Returns structured error response.
    """
    errors = []
    for err in exc.errors():
        loc = ".".join(str(l) for l in err["loc"])
        errors.append(f"{loc}: {err['msg']}")
    
    logger.warning(f"Validation error on {request.url.path}: {errors}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": {
                "status": "failed",
                "message": "Validation error",
                "errors": errors
            }
        },
    )


@api_app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(
    request: Request,
    exc: ValidationError
):
    """
    Handle Pydantic validation errors.
    Returns structured error response.
    """
    logger.warning(f"Pydantic validation error on {request.url.path}: {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": {
                "status": "failed",
                "message": "Invalid data format",
                "errors": exc.errors()
            }
        },
    )


@api_app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all unhandled exceptions.
    Logs error and returns generic error response.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True
    )
    
    # Don't expose internal error details in production
    error_message = str(exc) if FASTAPI_ENV != "production" else "An internal error occurred"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": {
                "status": "error",
                "message": error_message
            }
        },
    )


# Uvicorn entrypoint
if __name__ == "__main__":
    import uvicorn
    logger.info("About to start API")
    is_dev = FASTAPI_ENV == "development"

    uvicorn.run(
        "main:api_app",
        host="0.0.0.0",
        port=int(settings.PORT),
        reload=(FASTAPI_ENV == "development"),
        log_level=settings.LOG_LEVEL.lower(),
        workers=None if is_dev else 2,   # use 2 workers in non-dev
    )