import logging

# Импорты скриптов
from data.download_data import download_and_extract
from data.preprocess_data import process_all_datasets
from models_fitting.train_rf import train_for_all_datasets as train_rf_all
from models_fitting.train_catboost import train_for_all_datasets as train_catboost_all
from models_fitting.train_lstm import train_for_all_datasets as train_lstm_all
from models_fitting.train_cnn import train_for_all_datasets as train_cnn_all

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("=" * 80)
    logger.info("ЗАПУСК ML ПАЙПЛАЙНА")
    logger.info("=" * 80)

    # 1. Скачивание данных
    logger.info("Шаг 1: Скачивание данных...")
    try:
        url = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
        download_and_extract(url)
    except Exception as e:
        logger.error(f"❌ Ошибка скачивания: {e}")
        return
    
    # 2. Предобработка данных
    logger.info("Шаг 2: Предобработка данных...")
    try:
        process_all_datasets()
    except Exception as e:
        logger.error(f"❌ Ошибка предобработки: {e}")
        return
    
    # 3. Обучение моделей
    logger.info("Шаг 3: Обучение моделей...")

    models = [
        ("Random Forest", train_rf_all),
        ("CatBoost", train_catboost_all),
        ("LSTM", train_lstm_all),
        ("CNN", train_cnn_all)
    ]

    for name, train_func in models:
        logger.info(f"--- Обучение {name} ---")
        try:
            train_func()
        except Exception as e:
            logger.error(f"❌ Ошибка обучения {name}: {e}")

    logger.info("=" * 80)
    logger.info("ML ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()