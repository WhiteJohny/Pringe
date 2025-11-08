# DoubleAgent

DoubleAgent — интеллектуальный ассистент, объединяющий функционал работы с календарём и генерации шуток.

## Требования

- Python 3.8+
- Ollama
- Google Calendar API

### Установка зависимостей

```bash
pip install requests
pip install python-dateutil
pip install google-auth
pip install google-auth-oauthlib
pip install google-auth-httplib2
pip install google-api-python-client
```

## Настройка Ollama

### Установка Ollama

1. Перейдите на официальный сайт [Ollama](https://ollama.ai)
2. Скачайте и установите версию для вашей операционной системы
3. Запустите Ollama для активации сервиса

### Загрузка модели

Откройте терминал/командную строку и выполните команду:

```bash
ollama pull deepseek-r1:8b
```
Дождитесь завершения загрузки модели.

## Настройка Google Calendar API

### Создание проекта в Google Cloud Console

1. **Перейдите в Google Cloud Console**
   - Откройте [console.cloud.google.com](https://console.cloud.google.com)
   - Войдите в свой Google аккаунт

2. **Создайте новый проект**
   - Нажмите на выпадающий список проектов в верхней панели
   - Нажмите **"New Project"**
   - Введите название проекта (например, "DoubleAgent")
   - Нажмите **"Create"**

3. **Включите Google Calendar API**
   - В боковом меню перейдите в **"APIs & Services"** → **"Library"**
   - В поиске введите "Google Calendar API"
   - Выберите **"Google Calendar API"** из результатов
   - Нажмите кнопку **"Enable"**

### Создание учетных данных OAuth 2.0

4. **Настройка OAuth-согласия**
   - Перейдите в **"APIs & Services"** → **"OAuth consent screen"**
   - Выберите **"External"** тип пользователей
   - Заполните обязательные поля:
     - Название приложения: "DoubleAgent"
     - Email пользователя: ваш email
   - Нажмите **"Save and Continue"**

5. **Создание Client ID**
   - Перейдите в **"APIs & Services"** → **"Credentials"**
   - Нажмите **"Create Credentials"** → **"OAuth client ID"**
   - Выберите **"Desktop Application"** в качестве типа приложения
   - Введите название клиента (например, "DoubleAgent Desktop")
   - Нажмите **"Create"**

6. **Скачивание файла credentials.json**
   - После создания клиента появится всплывающее окно с данными
   - Нажмите кнопку **"Download JSON"**
   - Сохраните файл как `client_secret.json`
   - Поместите файл в корневую директорию вашего проекта

### Проверка настроек

Убедитесь, что в разделе **"Credentials"** отображается:
- ✅ Google Calendar API включен
- ✅ OAuth 2.0 Client ID создан
- ✅ Файл `client_secret.json` скачан и размещен в проекте
