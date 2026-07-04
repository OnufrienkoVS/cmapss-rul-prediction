from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class PredictionRequest(BaseModel):
    """Запрос на предсказание."""
    dataset: str = Field(..., description="Название датасета (FD001, FD002, FD003, FD004)")
    model_type: str = Field(..., description="Тип модели (catboost, rf, lstm, cnn)")
    data: List[List[float]] = Field(..., description="Входные данные (последовательность)")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset": "FD001",
                "model_type": "catboost",
                "data": [
                    [0.5, 1.2, 0.8, 0.1, 0.3, 0.5, 0.7, 0.2, 0.9, 0.4, 
                     0.6, 0.8, 0.3, 0.7, 0.5, 0.9, 0.2, 0.6, 0.4, 0.8, 
                     0.1, 0.5, 0.7, 0.3, 0.9, 0.2, 0.6, 0.8, 0.4, 0.7]
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Ответ с предсказанием."""
    rul: float = Field(..., description="Предсказанный остаточный ресурс в циклах")
    model_type: str = Field(..., description="Тип использованной модели")
    dataset: str = Field(..., description="Использованный датасет")
    status: str = Field(default="success", description="Статус выполнения")


class HealthResponse(BaseModel):
    """Ответ на проверку здоровья."""
    status: str = Field(default="healthy")
    models_available: List[str] = Field(..., description="Список доступных моделей")
    total_models: int = Field(..., description="Общее количество моделей")


class ModelsListResponse(BaseModel):
    """Список доступных моделей."""
    models: Dict[str, Dict] = Field(..., description="Словарь с данными моделей")


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""
    error: str
    detail: Optional[str] = None