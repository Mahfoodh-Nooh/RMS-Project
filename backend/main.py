from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine
import models
from routes import auth, portfolio, risk

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RMS SaaS API",
    description="Risk Management System — MVP Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(portfolio.router)
app.include_router(risk.router)


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "version": "1.0.0"}
