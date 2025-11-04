from fastapi import FastAPI
from app.routers import products, recognize

app = FastAPI(
    title="Product Intelligence API",
    description="An API for product recognition and catalog management",
    version="1.0.0"
)

# Include routers
app.include_router(products.router)
app.include_router(recognize.router)


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)