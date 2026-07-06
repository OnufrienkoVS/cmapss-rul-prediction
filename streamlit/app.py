import streamlit as st
import pandas as pd
import numpy as np
import requests

# Настройка страницы
st.set_page_config(
    page_title="RUL Prediction",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния сессии
if "data" not in st.session_state:
    st.session_state["data"] = None
if "data_source" not in st.session_state:
    st.session_state["data_source"] = None
if "use_random" not in st.session_state:
    st.session_state["use_random"] = False

# Заголовок
st.title(":material/monitor_heart: Прогнозирование остаточного ресурса (RUL)")
st.markdown("### Турбореактивные двигатели | NASA C-MAPSS")
st.markdown("---")

# Боковая панель
with st.sidebar:
    st.header(":material/settings: Параметры")
    
    # Выбор датасета
    dataset = st.selectbox(
        ":material/dataset: Датасет",
        ["FD001", "FD002", "FD003", "FD004"],
        help="Выберите датасет для предсказания"
    )
    
    # Выбор модели
    model_type = st.selectbox(
        ":material/model_training: Модель",
        ["catboost", "rf", "lstm", "cnn"],
        help="Выберите модель для предсказания"
    )
    
    st.markdown("---")
    st.markdown("### :material/upload_file: Загрузка данных")
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Загрузите CSV или TXT файл с данными",
        type=["csv", "txt"],
        help="CSV: данные в виде таблицы, TXT: сырые данные из датасета C-MAPSS"
    )
    
    # Генерация случайных данных
    st.markdown("---")
    st.markdown("### :material/casino: Или сгенерируйте пример")
    
    if st.button(":material/wand_stars: Случайные данные", use_container_width=True):
        # Генерируем данные (все 24 признака) и сохраняем состояние
        st.session_state["data"] = np.random.randn(30, 24).tolist()
        st.session_state["data_source"] = "random"
        st.session_state["use_random"] = True
        st.rerun()

    st.markdown("---")
    st.markdown("### :material/info: Информация")
    st.markdown(f"**Модель:** {model_type}")
    st.markdown(f"**Датасет:** {dataset}")
    
    # Количество признаков для каждого датасета
    feature_counts = {
        "FD001": 17,
        "FD002": 24,
        "FD003": 19,
        "FD004": 24
    }
    st.markdown(f"**Признаков:** {feature_counts.get(dataset, 24)}")
    st.markdown(f"**Окно:** 30 временных шагов")
    st.info(":material/notification_important: Для предсказания используется **последнее окно** из 30 шагов")

# Основная область
col1, col2 = st.columns([2, 1])

# Отображение данных
with col1:
    st.subheader(":material/data_table: Данные")
    
    # Загрузка из файла
    if uploaded_file is not None:
        try:
            # Определяем тип файла по расширению
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # CSV файл
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(10))
                st.caption(f"Всего строк: {len(df)}")
                
                if len(df) >= 30:
                    data = df.values[:30].tolist()
                    st.session_state["data"] = data
                    st.session_state["data_source"] = "csv"
                    st.caption(f"Всего строк: {len(df)}")
                    st.success(f"✅ Загружено {len(data)} временных шагов, {len(data[0]) if data else 0} признака")
                else:
                    st.warning(f"⚠️ Недостаточно данных. Нужно минимум 30 строк, загружено {len(df)}")
            
            elif file_extension == 'txt':
                content = uploaded_file.read().decode('utf-8')
                lines = content.strip().split('\n')
                
                parsed_data = []
                for line in lines:
                    values = [float(x) for x in line.split() if x.strip()]
                    if values:
                        parsed_data.append(values)
                
                if parsed_data:
                    if len(parsed_data) >= 30:
                        data = parsed_data[-30:]
                    else:
                        data = parsed_data
                    
                    st.session_state["data"] = data
                    st.session_state["data_source"] = "txt"

                    df = pd.DataFrame(data)
                    st.dataframe(df.head(10))
                    st.caption(f"Всего строк: {len(parsed_data)}")
                    st.success(f"✅ Загружено {len(data)} временных шагов, {len(data[0]) if data else 0} признака")
                else:
                    st.error("❌ Не удалось распарсить TXT файл")
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
    
    # Случайные данные
    elif st.session_state.get("data") is not None and st.session_state.get("data_source") == "random":
        data = st.session_state["data"]

        # Показываем данные
        df = pd.DataFrame(data)
        st.dataframe(df.head(10))
        st.caption(f"Всего строк: {len(df)}")
        st.info(f":material/casino: Сгенерировано {len(data)} случайных шагов, {len(data[0]) if data else 0} признака")
    
    else:
        st.info('Загрузите файл или сгенерируйте пример')
        data = None

with col2:
    st.subheader(":material/target: Предсказание")
    
    # Получаем данные из состояния
    data = st.session_state.get("data")
    data_source = st.session_state.get("data_source")
    
    # Кнопка предсказания
    if st.button(":material/play_arrow: Предсказать RUL", type="primary", use_container_width=True):
        if data is not None:
            with st.spinner("Загрузка модели и предсказание..."):
                try:
                    # Отправка запроса к API
                    response = requests.post(
                        "http://localhost:8000/api/v1/predict",
                        json={
                            "dataset": dataset,
                            "model_type": model_type,
                            "data": data
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        rul = result['rul']
                        
                        # Отображение результата
                        st.success("✅ Предсказание выполнено!")
                        st.metric(
                            label="Остаточный ресурс (RUL)",
                            value=f"{rul:.2f} циклов",
                            delta=None
                        )
                        
                        # Дополнительная информация
                        st.caption(f"Модель: {result['model_type']}")
                        st.caption(f"Датасет: {result['dataset']}")
                        st.caption(f"Статус: {result['status']}")
                        if data_source:
                            st.caption(f"Источник данных: {data_source}")
                        st.caption(":material/info: Использовано последнее окно из 30 шагов")
                        
                    else:
                        st.error(f"❌ Ошибка API: {response.status_code}")
                        try:
                            error_detail = response.json().get('detail', 'Неизвестная ошибка')
                            st.error(f"Детали: {error_detail}")
                        except:
                            st.error(response.text[:200])
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Не удалось подключиться к API. Убедитесь, что сервер запущен.")
                    st.info("Запустите API: `python run_api.py`")
                except requests.exceptions.Timeout:
                    st.error("❌ Превышено время ожидания ответа от API")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
        else:
            st.warning("⚠️ Сначала загрузите данные или сгенерируйте пример")

# Footer
st.markdown("---")
st.caption("🔬 Проект по прогнозированию остаточного ресурса (RUL) | NASA C-MAPSS | FastAPI + Streamlit")