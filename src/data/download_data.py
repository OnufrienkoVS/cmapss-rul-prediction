import requests
import zipfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def download_and_extract(url: str, extract_to='../../data/raw'):
    script_dir = Path(__file__).parent.absolute()

    extract_path = script_dir / Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)

    filename = 'CMAPSSData.zip'
    zip_path = extract_path / filename

    logger.info(f"Загрузка {filename} с {url}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Сохраняем файл с прогрессом
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Файл {filename} успешно загружен!")
        logger.info(f"Распаковка...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        logger.info(f"Распаковка завершена в: {extract_path.absolute()}")

        zip_path.unlink()
        
        
        print("\nСодержимое:")
        for item in extract_path.iterdir():
            if item.is_file():
                print(f"  - {item.name} ({item.stat().st_size} байт)")

    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    url = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
    download_and_extract(url)