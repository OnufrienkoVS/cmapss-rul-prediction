import yaml
import joblib
import logging
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


logger = logging.getLogger(__name__)


def load_best_params(config_path: str, dataset: str) -> dict:
    """Загружает параметры из конфиг файла."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config[dataset]['rf']


def train_and_save(dataset: str, config_path: str = '../../config/best_params.yaml',
                   data_dir: str = '../../data/processed', model_dir: str = '../../models') -> None:
    """Обучает и сохраняет модель."""
    logger.info(f"Обучение Random Forest на {dataset}")

    # Чтение параметров из конфига
    script_dir = Path(__file__).parent.absolute()
    abs_config_path = (script_dir / config_path).resolve()
    params = load_best_params(abs_config_path, dataset)
    logger.info(f"Параметры получены из {abs_config_path}")
    
    # Загрузка данных для обучения
    data_path = (script_dir / data_dir / dataset).resolve()
    X_train = pd.read_csv(data_path / 'X_train.csv')
    y_train = pd.read_csv(data_path / 'y_train.csv').squeeze()
    logger.info(f"Размер обучающих данных: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # Обучение модели
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        random_state=0,
        n_jobs=-1
    )
    model.fit(X_train.values, y_train.values)
    logger.info(f"Модель обучена!")
    
    # Сохранение модели
    model_path = (script_dir / model_dir / f'rf_{dataset}.pkl').resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Модель сохранена: {model_path}")


def train_for_all_datasets(
    config_path: str = '../../config/best_params.yaml',
    data_dir: str = '../../data/processed',
    model_dir: str = '../../models'
) -> None:
    """Обучает Random Forest модели для всех датасетов C-MAPSS."""
    print("=" * 100)
    logger.info("Запуск обучения rf на всех датасетах...")
    print("=" * 100)

    datasets = ['FD001', 'FD002', 'FD003', 'FD004']

    for dataset in datasets:
        try:
            train_and_save(dataset, config_path, data_dir, model_dir)
            logger.info(f"{dataset} - успешно обучен")
            print("-" * 100)
        except Exception as e:
            logger.error(f"❌ {dataset} - ошибка: {e}")
            continue


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_for_all_datasets()