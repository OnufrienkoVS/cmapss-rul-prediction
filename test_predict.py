import requests
import numpy as np

BASE_URL = "http://localhost:8000"


def test_predict(dataset="FD001", model_type="catboost"):
    """Тест предсказания с сырыми данными."""
    feature_counts = {
        "FD001": 17,
        "FD002": 24,
        "FD003": 19,
        "FD004": 24
    }
    n_features = feature_counts.get(dataset, 24)
    window_size = 30
    
    # Генерируем сырые данные
    raw_data = np.random.randn(window_size, n_features).tolist()
    
    payload = {
        "dataset": dataset,
        "model_type": model_type,
        "data": raw_data
    }
    
    print(f"Запрос: {model_type} на {dataset}")
    print(f"   Сырых признаков: {n_features}")
    print(f"   Окно: {window_size} шагов")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json=payload
    )
    
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ RUL: {result['rul']:.2f} циклов")
    else:
        print(f"   ❌ Ошибка: {response.text[:300]}...")
    
    return response


def test_health():
    """Проверка здоровья сервиса."""
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"Health: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Статус: {data['status']}")
        print(f"   Моделей доступно: {data['total_models']}")
        print(f"   Список: {data['models_available'][:3]}...")
    return response


if __name__ == "__main__":
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ API")
    print("=" * 50)
    
    try:
        # Проверка здоровья
        test_health()
        print()
        
        # Тест классических моделей
        print("--- Классические модели ---")
        test_predict("FD001", "catboost")
        print()
        test_predict("FD002", "rf")
        print()
        
        # DL модели (если есть сохраненные файлы)
        print("--- DL модели ---")
        test_predict("FD001", "lstm")
        print()
        test_predict("FD001", "cnn")
        
    except requests.exceptions.ConnectionError:
        print("❌ Сервер не запущен! Запустите: python run_api.py")
    except Exception as e:
        print(f"❌ Ошибка: {e}")