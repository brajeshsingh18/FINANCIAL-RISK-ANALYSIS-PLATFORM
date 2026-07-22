from fastapi import FastAPI
from api.routers.predict import router as predict_router
from api.routers.health import router as health_router

app=FastAPI(title="FINANCIAL RISK PLATFORM",version="1.0.0")

app.include_router(health_router)
app.include_router(predict_router)




