import joblib
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
from catboost import CatBoostRegressor

from app.core.config import settings
from app.core.models import get_model_class, LSTMModel, CNN1D


logger = logging.getLogger(__name__)


class ModelLoader:
    """Загрузчик моделей."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._available_models: Dict[str, Dict] = {}
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._scan_available_models()
    
    def _scan_available_models(self) -> None:
        """Сканирует папку models/ и собирает метаданные."""
        logger.info(f"Device (torch): {self._device}")
        logger.info("Сканирование доступных моделей...")
        
        for model_type in settings.AVAILABLE_MODEL_TYPES:
            for dataset in settings.AVAILABLE_DATASETS:
                model_key = f"{model_type}_{dataset}"
                model_path = settings.model_paths[model_type][dataset]
                
                if model_path.exists():
                    self._available_models[model_key] = {
                        "model_type": model_type,
                        "dataset": dataset,
                        "path": str(model_path),
                        "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
                        "loaded": False  # Пока не загружена
                    }
                    logger.debug(f"Найдена модель: {model_key}")
        
        logger.info(f"Доступно моделей: {len(self._available_models)}")
    
    def _load_model(self, model_key: str) -> Any:
        """Загружает конкретную модель в память."""
        if model_key not in self._available_models:
            raise ValueError(f"Модель {model_key} не найдена. Доступны: {list(self._available_models.keys())}")
        
        meta = self._available_models[model_key]
        model_path = Path(meta["path"])
        model_type = meta["model_type"]
        
        logger.info(f"Загрузка модели: {model_key} ({meta['size_mb']:.1f} MB)")
        
        try:
            if model_type == "catboost":
                model = CatBoostRegressor()
                model.load_model(str(model_path))
            
            elif model_type == "rf":
                model = joblib.load(model_path)
            
            elif model_type == "lstm":
                checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)
                dataset = meta.get('dataset', 'FD001')
                input_size = settings.get_feature_count(dataset)
                model = LSTMModel(input_size=input_size)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self._device)
                model.eval()
            
            elif model_type == "cnn":
                checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)
                dataset = meta.get('dataset', 'FD001')
                input_size = settings.get_feature_count(dataset)
                model = CNN1D(input_size=input_size, window_size=30)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self._device)
                model.eval()
            
            else:
                raise ValueError(f"Неизвестный тип модели: {model_type}")
            
            # Обновляем статус
            self._available_models[model_key]["loaded"] = True
            self._models[model_key] = model
            
            logger.info(f"✅ Модель загружена: {model_key}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {model_key}: {e}")
            raise
    
    def get_model(self, model_key: str) -> Any:
        """Возвращает модель, загружая её при необходимости."""
        # Если модель уже загружена — возвращаем
        if model_key in self._models:
            return self._models[model_key]
        
        # Если модель доступна, но не загружена — загружаем
        if model_key in self._available_models:
            return self._load_model(model_key)
        
        # Модель не найдена
        available = list(self._available_models.keys())
        raise ValueError(f"Модель {model_key} не найдена. Доступны: {available}")
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Возвращает список доступных моделей."""
        return self._available_models
    
    def is_loaded(self, model_key: str) -> bool:
        """Проверяет, загружена ли модель."""
        return model_key in self._models
    
    def unload_model(self, model_key: str) -> None:
        """Выгружает модель из памяти (если загружена)."""
        if model_key in self._models:
            del self._models[model_key]
            if model_key in self._available_models:
                self._available_models[model_key]["loaded"] = False
            logger.info(f"Модель выгружена: {model_key}")

    def extract_features_from_sequence(self, data: np.ndarray) -> np.ndarray:
        """
        Извлекает статистические признаки из последовательности.
        
        Args:
            data: 2D массив (window_size, n_features) - сырые данные
        
        Returns:
            1D массив признаков (mean, std, min, max, delta, last)
        """
        if data.ndim != 2:
            raise ValueError(f"Ожидается 2D массив, получен {data.ndim}D")

        features = []
        for col in range(data.shape[1]):
            col_data = data[:, col]
            features.extend([
                float(np.mean(col_data)),
                float(np.std(col_data)),
                float(np.min(col_data)),
                float(np.max(col_data)),
                float(col_data[-1] - col_data[0]),
                float(col_data[-1])
            ])
        return np.array(features, dtype=np.float32)
        
    def preprocess_for_classical_model(self, data: np.ndarray) -> np.ndarray:
        """
        Предобработка данных для классических моделей (CatBoost, Random Forest).
        
        Args:
            data: 3D массив (batch, window_size, n_features) или 2D (window_size, n_features)
        
        Returns:
            2D массив (batch, n_features * 6) с извлеченными признаками
        """
        # Если данные 2D, добавляем batch dimension
        if data.ndim == 2:
            data = data.reshape(1, *data.shape)
        elif data.ndim != 3:
            raise ValueError(f"Ожидается 2D или 3D массив, получен {data.ndim}D")
        
        # Извлекаем признаки для каждого элемента батча
        features_list = []
        for i in range(data.shape[0]):
            features = self.extract_features_from_sequence(data[i])
            features_list.append(features)
        
        return np.array(features_list, dtype=np.float32)
    
    def preprocess_for_dl(self, data: np.ndarray) -> torch.Tensor:
        """
        Предобработка данных для DL моделей.
        
        Args:
            data: 2D (window_size, n_features) или 3D (batch, window_size, n_features)
        
        Returns:
            torch.Tensor: (batch, window_size, n_features)
        """
        if data.ndim == 2:
            data = data.reshape(1, *data.shape)
        elif data.ndim != 3:
            raise ValueError(f"Ожидается 2D или 3D массив, получен {data.ndim}D")
        
        # Преобразуем в тензор
        tensor = torch.FloatTensor(data).to(self._device)
        return tensor
    
    def predict(self, model_key: str, data: np.ndarray) -> float:
        """Делает предсказание, автоматически загружая модель."""
        model = self.get_model(model_key)
        model_type = self._available_models[model_key]["model_type"]

        # Проверяем размерность данных
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 2:
            pass  # OK
        elif data.ndim == 3:
            pass  # OK
        else:
            raise ValueError(f"Ожидается 1D, 2D или 3D массив, получен {data.ndim}D")
        
        if model_type in ["catboost", "rf"]:
            processed_data = self.preprocess_for_classical_model(data)
            prediction = model.predict(processed_data)

        elif model_type in ["lstm", "cnn"]:
            tensor_data = self.preprocess_for_dl(data)
            with torch.no_grad():
                prediction = model(tensor_data)
                prediction = prediction.cpu().numpy()

        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
        
        if isinstance(prediction, np.ndarray):
            return float(prediction[0])
        return float(prediction)


model_loader = ModelLoader()