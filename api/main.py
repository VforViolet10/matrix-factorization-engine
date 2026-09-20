from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, factorization, applications


app = FastAPI(
    title="Matrix Factorization Engine API",
    description=(
        "API for matrix factorization and decomposition "
        "using the Matrix Factorization Engine."
    ),
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(factorization.router)
app.include_router(applications.router)


@app.get("/")
def root():
    return {
        "name": "Matrix Factorization Engine API",
        "version": "0.1.0",
        "status": "running",
    }
