from pathlib import Path
from typing import Dict, List

class Settings:
    # Корневая директория проекта
    BASE_DIR = Path(__file__).parent.parent.parent
    
    # Пути к данным и моделям
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data" / "processed"
    CONFIG_DIR = BASE_DIR / "config"
    
    # Датасеты
    AVAILABLE_DATASETS = ["FD001", "FD002", "FD003", "FD004"]
    
    # Модели
    AVAILABLE_MODEL_TYPES = ["catboost", "rf", "lstm", "cnn"]
    
    # Параметры для предобработки
    WINDOW_SIZE = 30
    RUL_THRESHOLD = 130
    
    # Колонки признаков
    OP_COLS = [f"op{i}" for i in range(1, 4)]
    SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
    
    # Сенсоры, которые были удалены для каждого датасета
    SENSORS_TO_DROP = {
        "FD001": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
        "FD002": [],
        "FD003": ["s1", "s5", "s16", "s18", "s19"],
        "FD004": []
    }
    
    @property
    def model_paths(self) -> Dict[str, Dict[str, Path]]:
        """Возвращает словарь с путями ко всем моделям."""
        paths = {}
        for model_type in self.AVAILABLE_MODEL_TYPES:
            paths[model_type] = {}
            for dataset in self.AVAILABLE_DATASETS:
                if model_type == "catboost" or model_type == "rf":
                    ext = "pkl"
                else:
                    ext = "pt"
                paths[model_type][dataset] = self.MODELS_DIR / f"{model_type}_{dataset}.{ext}"
        return paths
    
    def get_actual_feature_cols(self, dataset: str) -> List[str]:
        """Возвращает колонки признаков для конкретного датасета """
        sensor_cols = [col for col in self.SENSOR_COLS 
                      if col not in self.SENSORS_TO_DROP.get(dataset, [])]
        return self.OP_COLS + sensor_cols
    
    def get_feature_count(self, dataset: str) -> int:
        """Возвращает количество признаков для датасета."""
        return len(self.get_actual_feature_cols(dataset))
    
settings = Settings()

if __name__ == "__main__":
    print(f"BASE_DIR: {settings.BASE_DIR}")
    print(f"MODELS_DIR: {settings.MODELS_DIR}")
    print(f"DATA_DIR: {settings.DATA_DIR}")
    print(f"Model paths: {settings.model_paths}")