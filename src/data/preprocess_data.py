from pathlib import Path
import pandas as pd
import logging
from typing import Tuple, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CmapssPreprocessor:
    def __init__(self, dataset_name: str, raw_data_dir: str = '../../data/raw',
                 processed_data_dir: str = '../../data/processed', rul_threshold: int = 130,
                 var_threshold: float = 0.001, window_size: int = 30):
        script_dir = Path(__file__).parent.absolute()

        self.dataset_name = dataset_name
        self.raw_data_dir = (script_dir / raw_data_dir).resolve()
        self.processed_data_dir = (script_dir / processed_data_dir).resolve()
        self.rul_threshold = rul_threshold
        self.variance_threshold = var_threshold
        self.window_size = window_size

        self.col_names = ['unit', 'cycle'] + [f'op{i}' for i in range(1, 4)] + [f's{i}' for i in range(1, 22)]
        self.sensor_cols = [f's{i}' for i in range(1, 22)]
        self.op_cols = [f'op{i}' for i in range(1, 4)]
        self.feature_cols = self.op_cols + self.sensor_cols
        self.sensors_to_drop = None

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Processed data directory: {self.processed_data_dir}")

    def load_data(self, data_type: str = 'train') -> pd.DataFrame:
        """Загружает данные для указанного датасета."""
        if data_type == 'train':
            filename = f"train_{self.dataset_name}.txt"
        elif data_type == 'test':
            filename = f"test_{self.dataset_name}.txt"
        elif data_type == 'rul':
            filename = f"RUL_{self.dataset_name}.txt"
        else:
            raise ValueError(f"Неизвестный тип данных: {data_type}")
        
        file_path = self.raw_data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        if data_type == 'rul':
            df = pd.read_csv(file_path, sep=' ', header=None)
            if df.shape[1] > 1:
                df = df.iloc[:, 0]
            return df

        df = pd.read_csv(file_path, sep=' ', header=None, engine='python')
        if df.shape[1] > len(self.col_names):
            df = df.iloc[:, :len(self.col_names)]
        
        df.columns = self.col_names
        return df
    
    def identify_constant_sensors(self, train_df: pd.DataFrame) -> List[str]:
        """Определяет сенсоры с низкой информативностью (дисперсией)."""
        logger.info("Определение неинформативных сенсоров...")
        
        sensor_std_per_unit = train_df.groupby('unit')[self.sensor_cols].std()
        mean_std = sensor_std_per_unit.mean()
        sensors_to_drop = mean_std[mean_std < self.variance_threshold].index.tolist()
        
        logger.info(f"Найдено {len(sensors_to_drop)} неинформативных сенсоров: {sensors_to_drop}")
        
        return sensors_to_drop
    
    def add_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет колонку RUL и RUL clipped."""
        df['RUL'] = df.groupby('unit')['cycle'].transform('max') - df['cycle']
        df['RUL_clipped'] = df['RUL'].clip(upper=self.rul_threshold)
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Извлекает признаки из тренировочных данных."""
        logger.info("Извлечение признаков...")
        
        # Определяем актуальные колонки сенсоров
        actual_sensor_cols = [col for col in df.columns if col.startswith('s')]
        actual_feature_cols = self.op_cols + actual_sensor_cols
        
        all_features = []
        
        for unit_id, group in df.groupby('unit'):
            group = group.sort_values('cycle')
            
            for start in range(0, len(group) - self.window_size + 1):
                window = group.iloc[start:start + self.window_size]
                
                # Вычисляем статистики
                stats = []
                for col in actual_feature_cols:
                    stats.append(window[col].mean())
                    stats.append(window[col].std())
                    stats.append(window[col].min())
                    stats.append(window[col].max())
                    stats.append(window[col].iloc[-1] - window[col].iloc[0])
                    stats.append(window[col].iloc[-1])
                
                # Возьмём RUL в последний момент цикле окна
                rul = window['RUL_clipped'].iloc[-1]
                all_features.append(stats + [rul])
        
        # Создаем DataFrame
        stat_cols = []
        for col in actual_feature_cols:
            stat_cols += [
                f'{col}_mean',
                f'{col}_std',
                f'{col}_min',
                f'{col}_max',
                f'{col}_delta',
                f'{col}_last'
            ]
        
        columns = stat_cols + ['RUL']
        features_df = pd.DataFrame(all_features, columns=columns)
        
        X = features_df.drop(columns=['RUL'])
        y = features_df['RUL']
        
        return X, y
    
    def extract_features_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлекает признаки из тестовых данных.
        Для тестовых данных мы берем последнее окно для каждого двигателя.
        """
        logger.info("Извлечение признаков для тестовых данных...")
        
        actual_sensor_cols = [col for col in df.columns if col.startswith('s')]
        actual_feature_cols = self.op_cols + actual_sensor_cols
        
        all_features = []
        
        for unit_id, group in df.groupby('unit'):
            group = group.sort_values('cycle')
            
            # Берем последние window_size записей (если есть)
            if len(group) >= self.window_size:
                window = group.tail(self.window_size)
            else:
                window = group
            
            # Вычисляем статистики для каждого признака
            stats = []
            for col in actual_feature_cols:
                stats.append(window[col].mean())
                stats.append(window[col].std())
                stats.append(window[col].min())
                stats.append(window[col].max())
                stats.append(window[col].iloc[-1] - window[col].iloc[0])
                stats.append(window[col].iloc[-1])
            
            all_features.append(stats)
        
        # Создаем DataFrame
        stat_cols = []
        for col in actual_feature_cols:
            stat_cols += [
                f'{col}_mean',
                f'{col}_std',
                f'{col}_min',
                f'{col}_max',
                f'{col}_delta',
                f'{col}_last'
            ]
        
        X_test = pd.DataFrame(all_features, columns=stat_cols)
        
        return X_test
    
    def process_train_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Пайплайн обработки тренировочных данных."""
        logger.info(f"Обработка тренировочных данных для {self.dataset_name}...")
        
        # 1. Загрузка данных
        df = self.load_data('train')
        logger.info(f"Загружено {len(df)} записей")
        
        # 2. Определение и удаление неинформативных сенсоров
        self.sensors_to_drop = self.identify_constant_sensors(df)
        if self.sensors_to_drop:
            df = df.drop(columns=self.sensors_to_drop)
            logger.info(f"Удалены сенсоры: {self.sensors_to_drop}")
        
        # 3. Добавление RUL и признаков
        df = self.add_rul(df)
        X, y = self.extract_features(df)
        logger.info(f"Создано {len(X)} примеров для обучения")
        
        # Сохраняем обработанные данные
        self.save_processed_data(X, y, 'train')
        
        return X, y, self.sensors_to_drop
    
    def process_test_data(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Пайплайн обработки тестовых данных."""
        if self.sensors_to_drop is None:
            raise ValueError("Сначала нужно обработать тренировочные данные!")
        
        logger.info(f"Обработка тестовых данных для {self.dataset_name}...")
        
        # 1. Загрузка тестовых данных
        df_test = self.load_data('test')
        logger.info(f"Загружено {len(df_test)} записей")
        
        # 2. Удаление тех же сенсоров, что и train
        if self.sensors_to_drop:
            df_test = df_test.drop(columns=self.sensors_to_drop)
            logger.info(f"Удалены сенсоры: {self.sensors_to_drop}")
        
        # 3. Загрузка RUL (если доступно)
        try:
            y_test = self.load_data('rul')
            logger.info(f"Загружены реальные значения RUL для теста")
        except FileNotFoundError:
            logger.warning("Файл с RUL для теста не найден.")
            y_test = None
        
        # 4. Извлечение признаков
        X_test = self.extract_features_test(df_test)
        
        logger.info(f"Создано {len(X_test)} примеров для тестирования")
        
        # Сохраняем обработанные данные
        self.save_processed_data(X_test, y_test, 'test')
        
        return X_test, y_test
    
    def save_processed_data(self, X: pd.DataFrame, y: Optional[pd.Series], data_type: str):
        """Сохраняет обработанные данные."""
        save_dir = self.processed_data_dir / self.dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        X.to_csv(save_dir / f"X_{data_type}.csv", index=False)
        
        if y is not None:
            y.to_csv(save_dir / f"y_{data_type}.csv", index=False)
        
        # Сохраняем список удаленных сенсоров
        if self.sensors_to_drop is not None:
            pd.Series(self.sensors_to_drop).to_csv(
                save_dir / "sensors_to_drop.csv", index=False
            )
        
        logger.info(f"Данные сохранены в {save_dir}")

    def run_pipeline(self) -> Dict:
        """Прогон пайплайна обработки данных."""
        # Обработка train
        X_train, y_train, sensors_to_drop = self.process_train_data()
        # Обработка test
        X_test, y_test = self.process_test_data()
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'sensors_to_drop': sensors_to_drop,
            'dataset': self.dataset_name
        }
    
def process_all_datasets(
    datasets: List[str] = ['FD001', 'FD002', 'FD003', 'FD004'],
    raw_dir: str = "../../data/raw",
    processed_dir: str = "../../data/processed"
):
    """Обрабатывает все наборы данных."""
    results = {}
    
    for dataset in datasets:
        logger.info(f"=" * 60)
        logger.info(f"Обработка датасета {dataset}")
        logger.info(f"=" * 60)
        
        try:
            preprocessor = CmapssPreprocessor(
                dataset_name=dataset,
                raw_data_dir=raw_dir,
                processed_data_dir=processed_dir
            )
            results[dataset] = preprocessor.run_pipeline()
            logger.info(f"✅ {dataset} успешно обработан!\n")
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {dataset}: {e}")
            continue
    
    return results

if __name__ == "__main__":
    results = process_all_datasets()