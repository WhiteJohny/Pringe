from datetime import datetime

from calendar_agent import CalendarAgent
from joker_agent import JokerAgent


class DoubleAgent(CalendarAgent, JokerAgent):
    def __init__(self, model="deepseek-r1:8b"):
        super().__init__(model)
        self.model = model
        self.conversation_history = []
        self.max_history_length = 5
        self.system_prompt = self._build_system_prompt()
        self.joke_apis = {
            "jokeapi": "https://v2.jokeapi.dev/joke/Any?type=single&safe-mode",
            "official_joke": "https://official-joke-api.appspot.com/random_joke"
        }

    
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

            Также ты и веселый ассистент с доступом к базе шуток. Твои задачи:

            1. Если пользователь просит шутку, рассказывать анекдот или шутку
            2. Определять тему шутки по контексту (программирование, животные, наука и т.д.)
            3. В остальных случаях отвечать как обычный полезный ассистент

            Формат ответа со шуткой:
            "🎭 [Шутка]"

            Если шутка не сгенерирована тобой, а получена из API, можешь добавить комментарий.
        """
        return system_prompt

        
    
    def process_query(self, user_query):
        """Обрабатывает запрос пользователя, вызывает LLM и выполняет действие."""
        messages = self.get_contextual_messages(user_query)
        try:
            if self.detect_joke_request(user_query):
                api_joke = self.get_joke_from_api()

                response = f"🎭 Вот шутка для тебя:\n\n{api_joke}"
                query = f"Пользователь попросил шутку. Я уже получил шутку из API: '{api_joke}'."
                f"Добавь краткий веселый комментарий к этой шутке (1-2 предложения)."

                messages = self.get_contextual_messages(query)
                llm_comment = self._call_ollama(messages)['message']['content']
                
                result = response + llm_comment
                self._update_conversation_history(user_query, result)

                return f"{response}\n\n💬 {llm_comment}"
            
            response = self._call_ollama(messages)
            result = response['message']['content']
            self._update_conversation_history(user_query, result)

            action_data = self._extract_json_from_response(result)
            if not action_data:
                return result

            return self._execute_action(action_data)

        except Exception as e:
            return f"❌ Ошибка при обработке запроса: {str(e)}"


def main(agent):
    """Интерактивный чат с агентом."""
    print("🤖 Веселый & Календарный агент запущен! (для выхода введите 'выход' или 'quit')")
    print("Просто напиши что-нибудь, а если хочешь шутку - попроси!")
    print("Или")
    print("Доступные команды:")
    print("- 'покажи мои события'")
    print("- 'создай событие ...'")
    print("- 'какие встречи на неделе?'")
    print("- 'Поиск события название - события'")
    print("- 'удали событие id - события'")
    print()

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


# Запуск агента
if __name__ == "__main__":
    agent = DoubleAgent()
    main(agent)
