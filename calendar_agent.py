import os
import json
import re
import ollama

from datetime import datetime, timezone, timedelta
from dateutil import parser

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class TimeParser:
    def __init__(self):
        self.setup_patterns()
    
    
    def setup_patterns(self):
        """Настройка regex-паттернов для распознавания времени"""
        self.patterns = {
            # Относительные дни
            'today': r'(сегодня|сейчас|today|now)',
            'tomorrow': r'(завтра|tomorrow)',
            'after_tomorrow': r'(послезавтра|after tomorrow)',
            'yesterday': r'(вчера|yesterday)',
            
            # Дни недели
            'weekdays': {
                'понедельник': 0, 'вторник': 1, 'среду': 2, 'четверг': 3,
                'пятницу': 4, 'субботу': 5, 'воскресенье': 6,
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            },
            
            # Время суток
            'time_of_day': {
                'утром': (9, 0), 'утра': (9, 0), 'утро': (9, 0),
                'днем': (14, 0), 'дня': (14, 0),
                'вечером': (18, 0), 'вечера': (18, 0),
                'ночью': (22, 0), 'ночи': (22, 0),
                'morning': (9, 0), 'afternoon': (14, 0), 'evening': (18, 0)
            },
            
            # Через промежуток времени
            'in_duration': r'через\s+(\d+)\s*(час|часа|часов|минут|минуту|минуты|день|дня|дней|недел|неделю)',
            
            # Конкретное время
            'exact_time': r'(\d{1,2})[:\.]?(\d{2})?\s*(утра|вечера|дня|ночи|am|pm)?',
            
            # Даты
            'date_dmy': r'(\d{1,2})[\.\/\-](\d{1,2})(?:[\.\/\-](\d{2,4}))?',
            'date_text': r'(\d{1,2})\s*(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)'
        }
        
        self.months = {
            'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6,
            'июля': 7, 'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
        }


    def parse_natural_time(self, time_str, reference_time=None):
        """
        Парсит естественное описание времени
        
        Args:
            time_str: строка с описанием времени
            reference_time: опорное время (по умолчанию текущее)
        
        Returns:
            datetime объект в UTC
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        time_str_lower = time_str.lower().strip()
        result_time = reference_time.replace(second=0, microsecond=0)
        
        # 1. Проверяем абсолютное время (ISO формат)
        try:
            if 'T' in time_str or '-' in time_str:
                parsed = parser.isoparse(time_str)
                return parsed.astimezone() if parsed.tzinfo else parsed.replace()
        except (ValueError, TypeError):
            pass
        
        # 2. Обрабатываем относительные дни
        result_time = self._parse_relative_days(time_str_lower, result_time)
        
        # 3. Обрабатываем дни недели
        result_time = self._parse_weekdays(time_str_lower, result_time)
        
        # 4. Обрабатываем "через X времени"
        result_time = self._parse_duration(time_str_lower, result_time)
        
        # 5. Обрабатываем конкретное время
        result_time = self._parse_exact_time(time_str_lower, result_time)
        
        # 6. Обрабатываем даты
        result_time = self._parse_dates(time_str_lower, result_time)
        
        return result_time


    def _parse_relative_days(self, time_str, base_time):
        """Обработка относительных дней"""
        if re.search(self.patterns['today'], time_str):
            return base_time
        
        elif re.search(self.patterns['tomorrow'], time_str):
            return base_time + timedelta(days=1)
        
        elif re.search(self.patterns['after_tomorrow'], time_str):
            return base_time + timedelta(days=2)
        
        elif re.search(self.patterns['yesterday'], time_str):
            return base_time - timedelta(days=1)
        
        return base_time


    def _parse_weekdays(self, time_str, base_time):
        """Обработка дней недели"""
        for day_name, day_num in self.patterns['weekdays'].items():
            if day_name in time_str:
                current_weekday = base_time.weekday()
                days_ahead = day_num - current_weekday
                if days_ahead <= 0:
                    days_ahead += 7
                # Если указано "в следующий понедельник", добавляем неделю
                if 'следующ' in time_str or 'next' in time_str:
                    days_ahead += 7
                return base_time + timedelta(days=days_ahead)
        return base_time


    def _parse_duration(self, time_str, base_time):
        """Обработка промежутков времени"""
        duration_match = re.search(self.patterns['in_duration'], time_str)
        if duration_match:
            amount = int(duration_match.group(1))
            unit = duration_match.group(2)
            
            if unit in ['час', 'часа', 'часов', 'hour']:
                return base_time + timedelta(hours=amount)
            elif unit in ['минут', 'минуту', 'минуты', 'minute']:
                return base_time + timedelta(minutes=amount)
            elif unit in ['день', 'дня', 'дней', 'day']:
                return base_time + timedelta(days=amount)
            elif unit in ['недел', 'неделю', 'week']:
                return base_time + timedelta(weeks=amount)
        
        return base_time


    def _parse_exact_time(self, time_str, base_time):
        """Обработка конкретного времени"""
        time_match = re.search(self.patterns['exact_time'], time_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            period = time_match.group(3)
            
            # Корректируем 12-часовой формат
            if period in ['вечера', 'ночи', 'pm'] and hour < 12:
                hour += 12
            elif period in ['утра', 'дня', 'am'] and hour == 12:
                hour = 0
            
            return base_time.replace(hour=hour, minute=minute)
        
        # Проверяем время суток
        for time_of_day, (default_hour, default_minute) in self.patterns['time_of_day'].items():
            if time_of_day in time_str:
                return base_time.replace(hour=default_hour, minute=default_minute)
        
        return base_time


    def _parse_dates(self, time_str, base_time):
        """Обработка конкретных дат"""
        # Формат DD.MM или DD.MM.YYYY
        date_match = re.search(self.patterns['date_dmy'], time_str)
        if date_match:
            day = int(date_match.group(1))
            month = int(date_match.group(2))
            year = int(date_match.group(3)) if date_match.group(3) else base_time.year
            
            # Корректируем год если нужно
            if year < 100:
                year += 2000
            
            try:
                return base_time.replace(year=year, month=month, day=day)
            except ValueError:
                pass
        
        date_text_match = re.search(self.patterns['date_text'], time_str)
        if date_text_match:
            day = int(date_text_match.group(1))
            month_name = date_text_match.group(2)
            month = self.months.get(month_name)
            if month:
                try:
                    return base_time.replace(month=month, day=day)
                except ValueError:
                    pass
        
        return base_time


    def suggest_end_time(self, start_time, duration_hours=1):
        """Предлагает время окончания на основе времени начала"""
        return start_time + timedelta(hours=duration_hours)


class CalendarAgent:
    def __init__(self, model="deepseek-r1:8b"):
        self.model = model
        self.service = None
        self.conversation_history = []
        self.max_history_length = 10
        self.time_parser = TimeParser()
        self.setup_calendar_service()


    def setup_calendar_service(self):
        """Инициализация Google Calendar API."""
        SCOPES = ['https://www.googleapis.com/auth/calendar']
        creds = self._load_credentials(SCOPES)
        if not creds:
            creds = self._authenticate_user(SCOPES)
        self.service = build('calendar', 'v3', credentials=creds)
        print("✅ Google Calendar API подключен")


    def _load_credentials(self, scopes):
        """Загрузка сохранённых токенов."""
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', scopes)
            if creds.valid:
                return creds
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                return creds
            
        return None


    def _authenticate_user(self, scopes):
        """Аутентификация через OAuth."""
        flow = InstalledAppFlow.from_client_secrets_file(
            'client_secret.json',
            scopes
        )
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

        return creds
    

    def add_to_history(self, role, content):
        """Добавление сообщения в историю с ограничением длины."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[2:]


    def get_contextual_messages(self, user_query):
        """Формирует полный контекст для модели (system + история + запрос)."""
        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_query})

        return messages


    def _build_system_prompt(self):
        """Генерирует системный промпт с актуальными датой/временем."""
        now = datetime.now()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        day_of_week = now.strftime("%A")

        system_prompt = f"""
            Роль
            Ты — интеллектуальный ассистент с доступом к API календаря. Твоя задача — обрабатывать запросы пользователя, чётко разделяя:
            - операции с календарём (просмотр, создание, редактирование событий);
            - обычное общение.


            Текущее время и дата (используй для обработки запросов):
            - Дата: {current_date}
            - Время: {current_time}
            - День недели: {day_of_week}
            - Полное время: {current_datetime}

            Основные функции
            1. Показывать предстоящие события (с учётом текущей даты и времени).
            2. Создавать новые события в календаре (с валидацией времени).
            3. Искать события в календаре по названию.
            4. Удалять события по ID.
            4. Поддерживать естественный диалог по несвязанным с календарём темам.
            5. Показывать предстоящие события и брать оттуда информацию


            Правила обработки запросов
            1. Распознавание календарных интентов
            - Триггеры: «встреча», «событие», «напомни», «запланируй», «когда», «расписание», «свободное время» и аналогичные.
            - Для относительных временных указателей («завтра», «через неделю») конвертируй в абсолютные значения, используя текущую дату {current_date}.
            - При неоднозначности задавай уточняющий вопрос: «Вы хотите добавить событие в календарь?»

            2. Работа с датой и временем
            - Всегда используй актуальные значения выше для расчётов.
            - Если время не указано, уточни его у пользователя.
            - Указывай часовой пояс, если это критично.

            3. Формат ответа
            - Для календарных операций: возвращай JSON‑структуру (только при явном/неявном запросе к календарю).
            - Для обычного общения: отвечай текстом в естественном стиле.

            4. Обработка недостающих данных
            - Нет времени → «На какое время запланировать событие?»
            - Нет названия → «Как назовём это событие?»

            5. Если достаточно информации, не нужно переспрашивать о выполнении дейсвтия, если это не название, время начала и дата. Просто выполни его.

            6. Не нужно заставлять пользователя вводить какие-то команды, общайся с ним, как при живом общении.


            Формат JSON для календарных операций get_events|create_event|unknown"
            {{
                "action": "get_events|create_event|unknown",
                "parameters": {{
                    "max_results": 10,
                    "summary": "Название события (обязательно)",
                    "start_time": "YYYY-MM-DDTHH:MM:SS",
                    "end_time": "YYYY-MM-DDTHH:MM:SS",
                    "description": "Описание события (опционально)"
                }}
            }}
            Формат JSON для календарных операций search_events_by_title
            {{
                "action": "search_events_by_title",
                "parameters": {{
                    "title": "Название события (обязательно)",
                    "max_results": 10
                }}
            }}
            Формат JSON для календарных операций delete_event
            {{
                "action": "delete_event",
                "parameters": {{
                    "event_id": "id"
                }}
            }}
        """
        return system_prompt


    def get_events(self, max_results=10, days_ahead=7):
        """Получает ближайшие события."""
        try:
            time_min, time_max = self._get_time_range(days_ahead)
            events = self._fetch_calendar_events(time_min, time_max, max_results)
            return self._format_events_response(events)
        
        except Exception as e:
            return f"❌ Ошибка при получении событий: {str(e)}"


    def _get_time_range(self, days_ahead):
        """Возвращает timeMin и timeMax для запроса к календарю."""
        now = datetime.now(timezone.utc).isoformat()
        end_time = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat()

        return now, end_time


    def _fetch_calendar_events(self, time_min, time_max, max_results):
        """Запрашивает события у Google Calendar API."""
        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        return events_result.get('items', [])


    def _format_events_response(self, events):
        """Форматирует список событий в читаемый ответ."""
        if not events:
            return "На ближайшую неделю событий не найдено."
        
        result = "📅 Ваши ближайшие события:\n"
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            summary = event.get('summary', 'Без названия')
            formatted_time = self._format_event_time(start)
            result += f"• {formatted_time} - {summary}\n"

        return result


    def _format_event_time(self, start_str):
        """Конвертирует ISO‑время в читаемый формат."""
        if 'T' in start_str:
            dt = parser.isoparse(start_str)
            return dt.strftime("%d.%m.%Y %H:%M")
        else:
            dt = datetime.strptime(start_str, "%Y-%m-%d")
            return dt.strftime("%d.%m.%Y")


    def create_event(self, summary, start_time_str, end_time_str=None, description=""):
        """Создаёт событие в календаре."""
        try:
            start_dt = self.time_parser.parse_natural_time(start_time_str)
            end_dt = self._determine_end_time(start_dt, end_time_str)
            event_body = self._build_event_body(summary, start_dt, end_dt, description)
            created_event = self.service.events().insert(calendarId='primary', body=event_body).execute()

            return self._format_create_response(summary, start_dt, end_dt)
        
        except Exception as e:
            return f"❌ Ошибка при создании события: {str(e)}"


    def _determine_end_time(self, start_dt, end_time_str):
        """Определяет время окончания (из параметра или по умолчанию)."""
        if end_time_str:
            return self.time_parser.parse_natural_time(end_time_str, start_dt)
        else:
            return self.time_parser.suggest_end_time(start_dt)


    def _build_event_body(self, summary, start_dt, end_dt, description):
        """Формирует тело запроса для создания события."""
        return {
            'summary': summary,
            'description': description,
            'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'Europe/Moscow'},
            'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'Europe/Moscow'},
        }


    def _format_create_response(self, summary, start_dt, end_dt):
        """Формирует ответ после создания события."""
        start_formatted = start_dt.strftime('%d.%m.%Y %H:%M')
        end_formatted = end_dt.strftime('%d.%m.%Y %H:%M')

        return f"✅ Событие создано: '{summary}'\n📅 {start_formatted} - {end_formatted}"
    

    def search_events_by_title(self, title, calendar_id='primary', max_results=10):
        """Ищет события в календаре по подстроке в названии."""
        try:
            events = self._call_calendar_search(title, calendar_id, max_results)
            return self._format_search_results(events)
        
        except Exception as e:
            print(f"[ОШИБКА search_events_by_title] {str(e)}")
            return []


    def _call_calendar_search(self, title, calendar_id, max_results):
        """Выполняет запрос к API календаря для поиска событий."""
        events_result = self.service.events().list(
            calendarId=calendar_id,
            q=title,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        return events_result.get('items', [])


    def _format_search_results(self, events):
        """Форматирует найденные события в список словарей."""
        result = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            result.append({
                "id": event['id'],
                "summary": event.get('summary', 'Без названия'),
                "start_time": start,
                "end_time": end
            })

        return result


    def delete_event(self, event_id, calendar_id='primary'):
        """Удаляет событие из календаря по ID."""
        if not event_id:
            return {
                "success": False,
                "message": "Не указан ID события для удаления."
            }
        
        try:
            self.service.events().delete(
                calendarId=calendar_id,
                eventId=event_id
            ).execute()

            return {
                "success": True,
                "message": f"Событие с ID '{event_id}' успешно удалено."
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Ошибка при удалении события: {str(e)}"
            }


    def process_query(self, user_query):
        """Обрабатывает запрос пользователя, вызывает LLM и выполняет действие."""
        messages = self.get_contextual_messages(user_query)
        try:
            response = self._call_ollama(messages)
            result = response['message']['content']
            self._update_conversation_history(user_query, result)

            action_data = self._extract_json_from_response(result)
            if not action_data:
                return result

            return self._execute_action(action_data)

        except Exception as e:
            return f"❌ Ошибка при обработке запроса: {str(e)}"


    def _call_ollama(self, messages):
        """Отправляет запрос к Ollama и возвращает ответ."""
        return ollama.chat(
            model=self.model,
            messages=messages
        )


    def _update_conversation_history(self, user_query, response):
        """Обновляет историю диалога."""
        self.add_to_history("user", user_query)
        self.add_to_history("assistant", response)


    def _extract_json_from_response(self, response):
        """Извлекает JSON из ответа модели."""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            
        except json.JSONDecodeError:
            pass
        
        return


    def _execute_action(self, action_data):
        """Выполняет действие на основе JSON из ответа модели."""
        action = action_data.get("action", "unknown")
        params = action_data.get("parameters", {})

        if action == "get_events":
            max_results = params.get("max_results", 10)
            return self.get_events(max_results=max_results)


        elif action == "search_events_by_title":
            title = params.get("title", "")
            if not title:
                return "Укажите название для поиска событий."
            
            events = self.search_events_by_title(title)

            return self._format_events_list(events)

        elif action == "create_event":
            return self._handle_create_event(params)
        
        elif action == "delete_event":
            event_id = params.get("event_id")
            if not event_id:
                return "Для удаления укажите ID события в поле 'event_id'."
            
            result = self.delete_event(event_id)
            
            return result["message"]
        
        else:
            return action_data.get("response", "Неизвестное действие.")


    def _format_events_list(self, events):
        """Формирует читаемый список найденных событий."""
        if events:
            response = "Найденные события:\n"
            for ev in events:
                response += (f"- {ev['summary']} "
                           f"(ID: {ev['id']}, "
                           f"Начало: {ev['start_time']})\n")
                
            return response
        
        else:
            return "Не найдено событий с таким названием."


    def _handle_create_event(self, params):
        """Обрабатывает создание события по параметрам."""
        summary = params.get("summary", "")
        start_time = params.get("start_time", "")
        end_time = params.get("end_time", "")
        description = params.get("description", "")

        if not summary:
            return "Для создания события нужно указать название."
        
        if not start_time:
            return "Для создания события нужно указать время. На какое время запланировать?"

        return self.create_event(summary, start_time, end_time, description)


def main(agent):
    """Интерактивный чат с агентом."""
    print("🤖 Календарный агент запущен! (для выхода введите 'выход' или 'quit')")
    print("Доступные команды:")
    print("- 'покажи мои события'")
    print("- 'создай событие ...'")
    print("- 'какие встречи на неделе?'")
    print("- 'Поиск события название - события'")
    print("- 'удали событие id - события'")

    while True:
        try:
            user_input = input("👤 Вы: ").strip()
            if user_input.lower() in ['выход', 'quit', 'exit']:
                print("До свидания!")
                break
            if not user_input:
                continue

            print("🤖 Агент: ", end="")
            response = agent.process_query(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        
        except Exception as e:
            print(f"❌ Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    agent = CalendarAgent()
    main(agent)
