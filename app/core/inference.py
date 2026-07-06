import joblib
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.core.models import get_model_class, LSTMModel, CNN1D


logger = logging.getLogger(__name__)


class ModelLoader:
    """Загрузчик моделей."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._available_models: Dict[str, Dict] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._scan_available_models()

    def _get_expected_features(self, dataset: str) -> List[str]:
        """Возвращает список ожидаемых признаков для датасета."""
        return settings.get_actual_feature_cols(dataset)
    
    def _get_full_features(self) -> List[str]:
        """Возвращает полный список признаков (все сенсоры + операции)."""
        return settings.get_full_feature_cols()
    
    def _get_raw_cols(self) -> List[str]:
        """Возвращает полный список колонок сырых данных."""
        return settings.get_raw_cols()

    def _get_sensors_to_drop(self, dataset: str) -> List[str]:
        """Возвращает список сенсоров, которые нужно удалить для датасета."""
        return settings.SENSORS_TO_DROP.get(dataset, [])

    def _get_scaler(self, dataset: str) -> StandardScaler:
        """Загружает scaler для датасета."""
        if dataset in self._scalers:
            return self._scalers[dataset]
        
        scaler_path = (settings.SCALERS_DIR / f'{dataset}_scaler.pkl').resolve()
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            self._scalers[dataset] = scaler
            logger.info(f"✅ Scaler загружен для {dataset}")
            return scaler
        else:
            logger.warning(f"⚠️ Scaler не найден для {dataset}, используется StandardScaler по умолчанию")
            scaler = StandardScaler()
            self._scalers[dataset] = scaler
            return scaler
    
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

    def preprocess_input_data(self, data: np.ndarray, dataset: str) -> np.ndarray:
        """
        Предобработка входных данных:
        1. Проверяет количество признаков
            - Если кол-во признаков совпадает с датасетом - ничего не делает
            - Если все признаки - удаляет лишние сенсоры
            - Если признаков меньше - ошибка
        
        Args:
            data: 2D массив (window_size, n_features)
            dataset: Название датасета
        
        Returns:
            2D массив с правильным количеством признаков
        """
        if data.ndim != 2:
            raise ValueError(f"Ожидается 2D массив, получен {data.ndim}D")
        
        n_features = data.shape[1]
        expected_features = len(self._get_expected_features(dataset))
        full_features = len(self._get_full_features())
        raw_cols = len(self._get_raw_cols())
        
        logger.debug(f"Входные данные: {n_features} признаков, ожидается {expected_features}")
        
        # Ожидаемое кол-во признаков
        if n_features == expected_features:
            logger.debug("Количество признаков совпадает с ожидаемым")
            return data
        
        # Полный набор признаков (все сенсоры + операции)
        elif n_features == full_features:
            logger.info(f"Обнаружены все {full_features} признаков, удаляем лишние сенсоры...")
            
            # Получаем полный список признаков
            full_feature_names = self._get_full_features()
            expected_feature_names = self._get_expected_features(dataset)
            
            # Определяем индексы признаков, которые нужно оставить
            keep_indices = []
            for feature in expected_feature_names:
                keep_indices.append(full_feature_names.index(feature))
            
            # Оставляем только нужные признаки
            data_filtered = data[:, keep_indices]
            logger.info(f"Удалено {n_features - len(keep_indices)} признаков, осталось {len(keep_indices)}")
            
            return data_filtered
        
        # Сырые данные с unit и cycle (26 колонок)
        elif n_features == raw_cols:
            logger.info(f"Обнаружены сырые данные с {raw_cols} колонками (unit, cycle, op, sensors)")
            
            raw_col_names = self._get_raw_cols()
            expected_feature_names = self._get_expected_features(dataset)
            
            # Определяем индексы признаков, которые нужно оставить
            keep_indices = []
            for feature in expected_feature_names:
                keep_indices.append(raw_col_names.index(feature))
            
            data_filtered = data[:, keep_indices]
            logger.info(f"Удалено {n_features - len(keep_indices)} колонок (unit, cycle и лишние сенсоры), осталось {len(keep_indices)}")
            
            return data_filtered
        
        # Если количество признаков не совпадает ни с одним из вариантов
        else:
            raise ValueError(
                f"Неверное количество признаков: {n_features}. "
                f"Ожидается {expected_features} (для {dataset}) "
                f"или {full_features} (полный набор)."
            )
    
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
    
    def preprocess_for_dl(self, data: np.ndarray, dataset: str) -> torch.Tensor:
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
        
        # Получаем scaler для датасета
        scaler = self._get_scaler(dataset)

        normalized_batch = []
        for i in range(data.shape[0]):
            normalized = scaler.transform(data[i])
            normalized_batch.append(normalized)
        
        # Преобразуем в тензор
        tensor = torch.FloatTensor(np.array(normalized_batch)).to(self._device)
        return tensor
    
    def predict(self, model_key: str, data: np.ndarray) -> float:
        """Делает предсказание, автоматически загружая модель."""
        model = self.get_model(model_key)
        model_type = self._available_models[model_key]["model_type"]
        dataset = self._available_models[model_key]["dataset"]

        # Проверяем размерность данных
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 2:
            pass  # OK
        elif data.ndim == 3:
            pass  # OK
        else:
            raise ValueError(f"Ожидается 1D, 2D или 3D массив, получен {data.ndim}D")
        
        # Проверка и удаление лишних признаков
        if model_type in ["catboost", "rf"]:
            if data.ndim == 3:
                processed_batch = []
                for i in range(data.shape[0]):
                    processed = self.preprocess_input_data(data[i], dataset)
                    processed_batch.append(processed)
                data = np.array(processed_batch)
            else:
                data = self.preprocess_input_data(data, dataset)
        
        elif model_type in ["lstm", "cnn"]:
            if data.ndim == 3:
                processed_batch = []
                for i in range(data.shape[0]):
                    processed = self.preprocess_input_data(data[i], dataset)
                    processed_batch.append(processed)
                data = np.array(processed_batch)
            elif data.ndim == 2:
                data = self.preprocess_input_data(data, dataset)
                data = data.reshape(1, *data.shape)
            else:
                raise ValueError(f"Для DL моделей ожидается 2D или 3D массив")
        
        # Основная предобработка и инференс
        if model_type in ["catboost", "rf"]:
            processed_data = self.preprocess_for_classical_model(data)
            prediction = model.predict(processed_data)

            if isinstance(prediction, np.ndarray):
                return float(prediction[0])
            return float(prediction)

        elif model_type in ["lstm", "cnn"]:
            logger.debug(f"Data before DL preprocessing - min: {data.min():.4f}, max: {data.max():.4f}, mean: {data.mean():.4f}")
            tensor_data = self.preprocess_for_dl(data, dataset)
            logger.debug(f"Data after DL preprocessing - min: {tensor_data.min().item():.4f}, max: {tensor_data.max().item():.4f}, mean: {tensor_data.mean().item():.4f}")
            with torch.no_grad():
                prediction = model(tensor_data)
                prediction = prediction.cpu().numpy()

                prediction_flat = prediction.flatten()
                prediction_value = float(prediction_flat[0])

                return prediction_value

        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")


model_loader = ModelLoader()