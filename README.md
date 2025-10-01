# 🔍 Анализ тональности текста

CLI-приложение для анализа эмоциональной окраски текста на русском и английском языках.

## 🌟 Возможности

- Многоязычный анализ (русский, английский и др.)
- Подробные результаты с вероятностями
- Красивый интерфейс с эмодзи
- Работа в реальном времени

## 🚀 Быстрый старт

### Установка
```bash
git clone <ваш-репозиторий>
cd sentiment-analysis-cli
pip install -r requirements.txt
```

### Запуск
```bash
python main.py
```

## 💡 Примеры работы
```bash
📝 Enter text to analyze > I love this product!
⏳ Analyzing...

🎯 ANALYSIS RESULTS:
📝 Text: 'I love this product!'
🏆 Top emotion: 😊 positive (93.49%)

📊 DETAILED BREAKDOWN:
----------------------------------------
 1. 😊 positive    93.49%  (score: 0.9349)
 2. 😐 neutral      5.16%  (score: 0.0516)
 3. 😠 negative     1.36%  (score: 0.0136)
```

## 🛠 Команды

* Введите текст для анализа

* quit или q - выход

* help - справка

* clear - очистить экран

## 🏗 Технологии

* Модель: cardiffnlp/twitter-XLM-RoBERTa-base (многоязычная)

* Библиотеки: Hugging Face Transformers

* Тональности: Позитивная 😊, Нейтральная 😐, Негативная 😠
