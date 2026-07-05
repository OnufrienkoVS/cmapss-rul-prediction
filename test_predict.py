import requests
from pathlib import Path
import pandas as pd
from random import randint

BASE_URL = "http://localhost:8000"


def load_real_sequence(dataset="FD001", unit_id=1, window_size=30):
    """Загружает реальную последовательность из тренировочных данных."""
    # Загружаем сырые данные
    data_path = Path("data/raw") / f"train_{dataset}.txt"
    df = pd.read_csv(data_path, sep=' ', header=None)
    
    df = df.iloc[:, :26]
    df.columns = ['unit', 'cycle'] + [f'op{i}' for i in range(1, 4)] + [f's{i}' for i in range(1, 22)]
    
    sensors_to_drop = {
        "FD001": ["s1", "s5", "s6", "s10", "s16", "s18", "s19"],
        "FD002": [],
        "FD003": ["s1", "s5", "s16", "s18", "s19"],
        "FD004": []
    }
    drop_cols = sensors_to_drop.get(dataset, [])
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Выбираем двигатель
    unit_data = df[df['unit'] == unit_id].sort_values('cycle')
    len_data = len(unit_data)
    
    if len_data < window_size:
        raise ValueError(f"Двигатель {unit_id} имеет только {len(unit_data)} записей, нужно {window_size}")
    
    start = randint(0, len_data - window_size)
    
    sequence = unit_data.iloc[start:start + window_size].copy()

    max_cycle = unit_data['cycle'].max()
    sequence['rul'] = max_cycle - sequence['cycle']
    
    # Берем только признаки (op + sensors)
    actual_sensor_cols = [col for col in sequence.columns if col.startswith('s')]
    feature_cols = ['op1', 'op2', 'op3'] + actual_sensor_cols
    data = sequence[feature_cols].values.tolist()

    true_rul = sequence['rul'].iloc[-1]
    
    return data, true_rul

def test_with_real_data(dataset="FD001", model_type="catboost", unit_id=1):
    """Тестирует предсказание на реальных данных."""
    try:
        data, true_rul = load_real_sequence(dataset, unit_id)
        n_features = len(data[0])
        window_size = len(data)
        
        payload = {
            "dataset": dataset,
            "model_type": model_type,
            "data": data
        }
        
        print(f"Запрос: {model_type} на {dataset}, двигатель {unit_id}")
        print(f"   Реальное RUL: {true_rul:.2f} циклов")
        print(f"   Данные: {window_size} шагов, {n_features} признаков")
        
        response = requests.post(
            f"{BASE_URL}/api/v1/predict",
            json=payload
        )
        
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            predicted_rul = result['rul']
            error = abs(predicted_rul - true_rul)
            print(f"   ✅ Предсказанный RUL: {predicted_rul:.2f} циклов")
            print(f"   Ошибка: {error:.2f} циклов")

        else:
            print(f"   ❌ Ошибка: {response.text[:300]}...")
        
        return response
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

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
        test_with_real_data("FD001", "catboost", 1)
        print()
        test_with_real_data("FD002", "rf", 1)
        print()
        
        # DL модели (если есть сохраненные файлы)
        print("--- DL модели ---")
        test_with_real_data("FD001", "lstm", 1)
        print()
        test_with_real_data("FD001", "cnn", 1)
        print()
        
    except requests.exceptions.ConnectionError:
        print("❌ Сервер не запущен! Запустите: python run_api.py")
    except Exception as e:
        print(f"❌ Ошибка: {e}")