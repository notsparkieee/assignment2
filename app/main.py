"""
FastAPI Application Entry Point

This is the main application file that:
- Creates the FastAPI instance
- Configures middleware
- Registers routes
- Sets up health checks
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.vector_routes import router as vector_router
from app.config import settings
from app.exceptions import VectorServiceError

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready vector database with semantic search and metadata filtering",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS (Cross-Origin Resource Sharing)
# Why CORS? Allows frontend applications to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    Root endpoint - basic API information
    
    Why needed? Quick way to verify API is running
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Why needed?
    - Docker healthchecks
    - Load balancer health probes
    - Monitoring systems
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


app.include_router(vector_router, prefix="/vector", tags=["Vector Operations"])


@app.exception_handler(VectorServiceError)
async def vector_service_error_handler(
    _request: Request,
    exc: VectorServiceError,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "status_code": exc.status_code},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
