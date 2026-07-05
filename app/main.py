from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = FastAPI(
    title="C-MAPSS RUL Prediction API",
    description=(
        "API для прогнозирования остаточного ресурса (RUL) "
        "турбореактивных двигателей на основе данных NASA C-MAPSS.\n\n"
        "Доступные модели: CatBoost, Random Forest, LSTM, 1D-CNN\n"
        "Доступные датасеты: FD001, FD002, FD003, FD004"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем маршруты
app.include_router(routes.router)

# Корневой эндпоинт
@app.get("/", tags=["Root"])
async def root():
    """Корневой эндпоинт с информацией о сервисе."""
    return {
        "service": "C-MAPSS RUL Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health",
        "models": "/api/v1/models"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )