from fastapi import APIRouter
from fastapi import HTTPException
from api.schemas.request import PredictionInputSchema
from core_logic.inference.predict import analyse_stock

router=APIRouter(prefix="/predict",tags=["Prediction"])

@router.post("/")
def predict(request:PredictionInputSchema):
    try:
        result = analyse_stock(request.company)
        return result
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
