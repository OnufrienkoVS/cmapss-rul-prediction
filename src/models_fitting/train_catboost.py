import yaml
import logging
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor


logger = logging.getLogger(__name__)


def load_best_params(config_path: str, dataset: str) -> dict:
    """Загружает лучшие параметры из конфигурационного файла."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config[dataset]['catboost']


def train_and_save(dataset: str, config_path: str = '../../config/best_params.yaml',
                   data_dir: str = '../../data/processed', model_dir: str = '../../models') -> None:
    """Обучает и сохраняет модель."""
    logger.info(f"Обучение CatBoost на {dataset}")

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
    model = CatBoostRegressor(
        iterations=params['iterations'],
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        border_count=params['border_count'],
        random_seed=0,
        verbose=False,
        allow_writing_files=False
    )
    model.fit(X_train, y_train)
    logger.info(f"Модель обучена!")
    
    # Сохранение модели
    model_path = (script_dir / model_dir / f'catboost_{dataset}.pkl').resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info(f"Модель сохранена: {model_path}")


def train_for_all_datasets(
    config_path: str = '../../config/best_params.yaml',
    data_dir: str = '../../data/processed',
    model_dir: str = '../../models'
) -> None:
    """Обучает CatBoost модели для всех датасетов C-MAPSS."""
    print("=" * 100)
    logger.info("Запуск обучения CatBoost на всех датасетах...")
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