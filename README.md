# 🔍 Анализ тональности текста

API для анализа эмоциональной окраски текста на русском и английском языках.

## 🌟 Возможности

- Многоязычный анализ (русский, английский и др.)
- Подробные результаты с вероятностями
- Красивый интерфейс с эмодзи
- Работа в реальном времени

## 🚀 Быстрый старт

### Установка
```bash
git clone -b sentiment-analysis-api https://github.com/WhiteJohny/Pringe sentiment-analysis-api
cd sentiment-analysis-api
pip install -r requirements.txt
```

### Запуск
```bash
fastapi dev main.py
```

### Endpoints
Список можно найти в корне API: http://localhost:8000/ \
Или в документации: http://localhost:8000/docs

## 🏗 Технологии

* Модель: cardiffnlp/twitter-XLM-RoBERTa-base (многоязычная)

* Библиотеки: Hugging Face Transformers

* Тональности: Позитивная 😊, Нейтральная 😐, Негативная 😠
