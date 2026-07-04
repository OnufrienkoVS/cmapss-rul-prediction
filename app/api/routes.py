from fastapi import APIRouter, HTTPException, status
# from fastapi.responses import JSONResponse
from typing import List
import numpy as np
import logging

from app.api.models import (
    PredictionRequest, 
    PredictionResponse, 
    HealthResponse,
    ModelsListResponse,
    ErrorResponse
)
from app.core.inference import model_loader
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["API v1"])

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса"
)
async def health_check():
    """Проверяет, что сервис работает."""
    models = model_loader.get_available_models()
    return HealthResponse(
        status="healthy",
        models_available=list(models.keys()),
        total_models=len(models)
    )


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="Список доступных моделей"
)
async def list_models():
    """Возвращает список всех загруженных моделей с метаданными."""
    return ModelsListResponse(models=model_loader.get_available_models())


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Предсказание RUL"
)
async def predict(request: PredictionRequest):
    """
    Предсказывает остаточный ресурс на основе входных данных.
    
    - **dataset**: Название датасета (FD001, FD002, FD003, FD004)
    - **model_type**: Тип модели (catboost, rf, lstm, cnn)
    - **data**: Последовательность данных сенсоров (2D массив)
    """
    try:
        # Проверка датасета
        if request.dataset not in settings.AVAILABLE_DATASETS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неизвестный датасет. Доступны: {settings.AVAILABLE_DATASETS}"
            )
        
        # Проверяем, что модель существует
        if request.model_type not in settings.AVAILABLE_MODEL_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неизвестный тип модели. Доступны: {settings.AVAILABLE_MODEL_TYPES}"
            )
        
        model_key = f"{request.model_type}_{request.dataset}"
        
        # Проверяем, что модель доступна
        available_models = model_loader.get_available_models()
        if model_key not in available_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Модель {model_key} не найдена. Доступны: {list(available_models.keys())}"
            )
        
        # Преобразуем данные в numpy
        try:
            data = np.array(request.data, dtype=np.float32)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ошибка преобразования данных: {str(e)}"
            )
        
        # Проверяем, что данные не пустые
        if data.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Данные не могут быть пустыми"
            )
        
        # Предсказание
        rul = model_loader.predict(model_key, data)
        
        return PredictionResponse(
            rul=float(rul),
            model_type=request.model_type,
            dataset=request.dataset
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    

@router.post(
    "/predict_batch",
    response_model=List[PredictionResponse],
    summary="Пакетное предсказание"
)
async def predict_batch(requests: List[PredictionRequest]):
    """Делает пакетное предсказание для нескольких запросов."""
    results = []
    for request in requests:
        try:
            response = await predict(request)
            results.append(response)
        except HTTPException as e:
            results.append({
                "error": e.detail,
                "dataset": request.dataset,
                "model_type": request.model_type,
                "rul": -1,
                "status": "error"
            })
    return results