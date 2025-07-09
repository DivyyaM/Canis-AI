from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Canis AI Backend",
    description="Advanced AutoML Backend with Real-time Inference, Model Versioning, and Monitoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Canis AI Backend is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/healthcheck"
    }