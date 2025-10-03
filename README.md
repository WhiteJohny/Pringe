# 🔍 Анализ тональности текста

CLI-приложение для взаимодействия с LLM на русском и английском языках.

## 🌟 Возможности

- Многоязычное распознавание текста (русский, английский и др.)
- Красивый интерфейс с эмодзи
- Работа в реальном времени

## 🚀 Быстрый старт

### Установка
```bash
git clone <репозиторий>
cd sentiment-analysis-cli
pip install -r requirements.txt
```

### Аппаратное ускорение (GPU)
- Перейти на сайт [PyTorch](https://pytorch.org/get-started/locally/)
- Указать системные параметры (OS, версия CUDA)
- Скопировать и выполнить команду

Пример команды (Windows, CUDA 12.9):
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

### Запуск
```bash
python main.py
```

## 💡 Примеры работы
```
📝 Enter your message > Tell me a joke
⏳ Thinking...
user
Tell me a joke


Why don't scientists trust atoms? 

Because they make up everything! 

```

## 🛠 Команды

* Введите текст запроса

* quit или q - выход

* help - справка

* clear - очистить экран

## 🏗 Технологии

* Модель: Gemma 2 2B ([google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it))

* Библиотеки: Hugging Face Transformers
